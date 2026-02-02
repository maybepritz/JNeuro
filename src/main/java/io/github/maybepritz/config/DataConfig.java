package io.github.maybepritz.config;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class DataConfig {
    public String encoding = "UTF-8";
    public String delimiter = ";";
    public boolean useHeaders = false;
    public int minColumns = 1;
    public boolean verbose = true;
    public List<FieldMapping> fields = new ArrayList<>();
    public Map<String, Function<Object, Boolean>> filters = new HashMap<>();

    public DataConfig addField(FieldMapping field) {
        fields.add(field);
        minColumns = Math.max(minColumns, field.columnIndex + 1);
        return this;
    }

    public DataConfig addFilter(String fieldName, Function<Object, Boolean> filter) {
        filters.put(fieldName, filter);
        return this;
    }

    public FieldMapping getField(String name) {
        return fields.stream()
                .filter(f -> f.name.equals(name))
                .findFirst()
                .orElse(null);
    }
}