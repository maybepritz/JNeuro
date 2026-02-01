package io.github.maybepritz.modelio;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import io.github.maybepritz.activations.ActivationFunction;
import io.github.maybepritz.config.LayerConfig;
import io.github.maybepritz.config.NetworkConfig;
import io.github.maybepritz.layers.DenseLayer;
import io.github.maybepritz.layers.Layer;
import io.github.maybepritz.layers.SoftmaxLayer;
import io.github.maybepritz.core.NeuralNetwork;
import io.github.maybepritz.utils.Matrix;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

public class NetworkIO {
    private static final Gson gson = new GsonBuilder().setPrettyPrinting().create();
    private static final String CONFIG_FILE = "network_config.json";
    private static final String WEIGHTS_FILE = "network_coefficients.bin";
    private static final String METADATA_FILE = "network_meta.json";
    private static final String FORMAT_VERSION = "1.0";

    /**
     * Сохраняет нейронную сеть в ZIP-архив
     */
    public static void save(NeuralNetwork network, String path) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(path);
             ZipOutputStream zos = new ZipOutputStream(fos)) {

            // Конфиг сети и слои
            NetworkFormat.Config configData = serializeConfig(network.getConfig());
            List<NetworkFormat.LayerMeta> layersMeta = new ArrayList<>();
            for (Layer layer : network.getLayers()) {
                layersMeta.add(extractLayerMeta(layer));
            }
            NetworkFormat.Snapshot snapshot = new NetworkFormat.Snapshot();
            snapshot.config = configData;
            snapshot.layers = layersMeta;
            writeJsonToZip(zos, CONFIG_FILE, snapshot);

            // Веса (бинарно)
            writeWeightsToZip(zos, network);

            // Метаданные
            NetworkFormat.Meta meta = new NetworkFormat.Meta();
            meta.version = FORMAT_VERSION;
            meta.framework = "maybepritz-ann";
            meta.timestamp = System.currentTimeMillis();
            writeJsonToZip(zos, METADATA_FILE, meta);
        }
    }

    /**
     * Загружает нейронную сеть из ZIP-архива
     */
    public static NeuralNetwork load(String path) throws Exception {
        NetworkFormat.Snapshot snapshot = null;
        NetworkFormat.Meta meta = null;
        byte[] weightsData = null;

        try (FileInputStream fis = new FileInputStream(path);
             ZipInputStream zis = new ZipInputStream(fis)) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                String name = entry.getName();
                if (CONFIG_FILE.equals(name)) {
                    snapshot = gson.fromJson(new InputStreamReader(zis), NetworkFormat.Snapshot.class);
                } else if (WEIGHTS_FILE.equals(name)) {
                    weightsData = readAllBytes(zis);
                } else if (METADATA_FILE.equals(name)) {
                    meta = gson.fromJson(new InputStreamReader(zis), NetworkFormat.Meta.class);
                }
                zis.closeEntry();
            }
        }

        if (snapshot == null) {
            throw new IOException("Некорректный файл модели: отсутствует конфигурация");
        }

        // Проверка совместимости версии
        if (meta != null && !isVersionCompatible(meta.version)) {
            throw new IOException("Несовместимая версия модели: " + meta.version);
        }

        NetworkConfig netConfig = deserializeConfig(snapshot.config);
        NeuralNetwork network = new NeuralNetwork(netConfig);

        // Создаём слои
        for (NetworkFormat.LayerMeta layerMeta : snapshot.layers) {
            Layer layer = createLayerFromMeta(layerMeta, netConfig);
            network.addLayer(layer);
        }

        // Загружаем веса (если есть)
        if (weightsData != null && weightsData.length > 0) {
            loadWeightsFromBinary(network, weightsData);
        }

        return network;
    }

    /**
     * Сохраняет только веса сети
     */
    public static void saveWeightsOnly(NeuralNetwork network, String path) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(path);
             DataOutputStream dos = new DataOutputStream(fos)) {
            for (Layer layer : network.getLayers()) {
                if (layer.isTrainable() && layer.getWeights() != null) {
                    writeMatrixBinary(dos, layer.getWeights());
                    if (layer.getBias() != null) {
                        writeMatrixBinary(dos, layer.getBias());
                    }
                }
            }
        }
    }

    /**
     * Загружает только веса в существующую сеть
     */
    public static void loadWeightsOnly(NeuralNetwork network, String path) throws IOException {
        try (FileInputStream fis = new FileInputStream(path);
             DataInputStream dis = new DataInputStream(fis)) {
            for (Layer layer : network.getLayers()) {
                if (layer.isTrainable() && layer.getWeights() != null) {
                    readMatrixBinary(dis, layer.getWeights());
                    if (layer.getBias() != null) {
                        readMatrixBinary(dis, layer.getBias());
                    }
                }
            }
        }
    }

    // ==================== PRIVATE METHODS ====================

    private static boolean isVersionCompatible(String version) {
        return version != null && version.startsWith("1.");
    }

    private static void writeWeightsToZip(ZipOutputStream zos, NeuralNetwork network) throws IOException {
        zos.putNextEntry(new ZipEntry(WEIGHTS_FILE));
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (DataOutputStream dos = new DataOutputStream(baos)) {
            for (Layer layer : network.getLayers()) {
                if (layer.isTrainable() && layer.getWeights() != null) {
                    writeMatrixBinary(dos, layer.getWeights());
                    if (layer.getBias() != null) {
                        writeMatrixBinary(dos, layer.getBias());
                    }
                }
            }
        }
        zos.write(baos.toByteArray());
        zos.closeEntry();
    }

    private static void loadWeightsFromBinary(NeuralNetwork network, byte[] data) throws IOException {
        ByteArrayInputStream bais = new ByteArrayInputStream(data);
        DataInputStream dis = new DataInputStream(bais);
        for (Layer layer : network.getLayers()) {
            if (layer.isTrainable() && layer.getWeights() != null) {
                readMatrixBinary(dis, layer.getWeights());
                if (layer.getBias() != null) {
                    readMatrixBinary(dis, layer.getBias());
                }
            }
        }
    }

    private static void writeMatrixBinary(DataOutputStream dos, Matrix matrix) throws IOException {
        int rows = matrix.getRows();
        int cols = matrix.getCols();
        dos.writeInt(rows);
        dos.writeInt(cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dos.writeDouble(matrix.get(i, j));
            }
        }
    }

    private static void readMatrixBinary(DataInputStream dis, Matrix matrix) throws IOException {
        int rows = dis.readInt();
        int cols = dis.readInt();
        if (rows != matrix.getRows() || cols != matrix.getCols()) {
            throw new IOException("Размеры матрицы не совпадают: ожидалось " +
                    matrix.getRows() + "x" + matrix.getCols() +
                    ", получено " + rows + "x" + cols);
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix.set(i, j, dis.readDouble());
            }
        }
    }

    private static void writeJsonToZip(ZipOutputStream zos, String filename, Object data) throws IOException {
        zos.putNextEntry(new ZipEntry(filename));
        String json = gson.toJson(data);
        zos.write(json.getBytes(StandardCharsets.UTF_8));
        zos.closeEntry();
    }

    private static byte[] readAllBytes(InputStream is) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int read;
        while ((read = is.read(buffer)) != -1) {
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

    private static NetworkFormat.Config serializeConfig(NetworkConfig config) {
        NetworkFormat.Config data = new NetworkFormat.Config();
        data.learningRate = config.getLearningRate();
        data.l2Regularization = config.getL2Regularization();
        data.dropoutRate = config.getDropoutRate();
        data.weightInit = config.getWeightInit().name();
        data.gradientClip = config.getGradientClip();
        data.seed = config.getSeed();
        return data;
    }

    private static NetworkConfig deserializeConfig(NetworkFormat.Config data) {
        NetworkConfig config = new NetworkConfig();
        config.setLearningRate(data.learningRate);
        config.setL2Regularization(data.l2Regularization);
        config.setDropoutRate(data.dropoutRate);
        config.setWeightInit(NetworkConfig.WeightInit.valueOf(data.weightInit));
        config.setGradientClip(data.gradientClip);
        config.setSeed(data.seed);
        return config;
    }

    /**
     * Извлекает метаданные слоя для сериализации
     */
    private static NetworkFormat.LayerMeta extractLayerMeta(Layer layer) {
        NetworkFormat.LayerMeta meta = new NetworkFormat.LayerMeta();

        // Определяем тип слоя
        meta.layerType = layer.getClass().getSimpleName();

        // Для обучаемых слоёв сохраняем параметры
        Matrix weights = layer.getWeights();
        if (weights != null) {
            meta.inputSize = weights.getCols();
            meta.outputSize = weights.getRows();
        }

        ActivationFunction activation = layer.getActivation();
        if (activation != null) {
            meta.activationClass = activation.getClass().getName();
        }

        meta.useBias = layer.getBias() != null;

        return meta;
    }

    /**
     * Создаёт слой из метаданных
     */
    private static Layer createLayerFromMeta(NetworkFormat.LayerMeta meta, NetworkConfig netConfig) throws Exception {
        return switch (meta.layerType) {
            case "DenseLayer" -> createDenseLayer(meta, netConfig);
            case "SoftmaxLayer" -> new SoftmaxLayer();
            // Добавляйте новые типы слоёв здесь:
            // case "BatchNormLayer" -> new BatchNormLayer(...);
            // case "DropoutLayer" -> new DropoutLayer(...);
            default -> throw new IOException("Неизвестный тип слоя: " + meta.layerType);
        };
    }

    private static DenseLayer createDenseLayer(NetworkFormat.LayerMeta meta, NetworkConfig netConfig) throws Exception {
        if (meta.inputSize == null || meta.outputSize == null) {
            throw new IOException("DenseLayer требует inputSize и outputSize");
        }
        if (meta.activationClass == null) {
            throw new IOException("DenseLayer требует функцию активации");
        }

        Class<?> activationClass = Class.forName(meta.activationClass);
        ActivationFunction activation = (ActivationFunction) activationClass.getDeclaredConstructor().newInstance();

        LayerConfig layerConfig = new LayerConfig(meta.inputSize, meta.outputSize, activation);
        layerConfig.setUseBias(meta.useBias != null && meta.useBias);

        if (meta.dropoutRate != null) {
            layerConfig.setDropoutRate(meta.dropoutRate);
        }

        return new DenseLayer(layerConfig, netConfig);
    }
}