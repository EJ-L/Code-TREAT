package src;

import java.lang.*;
import java.util.*;
import java.util.stream.*;
import java.util.regex.*;
import java.security.*;
import java.io.FileWriter;
import java.math.*;
import java.io.*;
import java.nio.charset.*;
import static src.Utils.myStringify;
import static src.Global.*;

class Utils {
    private static String escapeString(String s) {
        StringBuilder newS = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch(c) {
                case '\\':
                    newS.append("\\\\");
                    break;
                case '\"':
                    newS.append("\\\"");
                    break;
                case '\n':
                    newS.append("\\n");
                    break;
                case '\t':
                    newS.append("\\t");
                    break;
                case '\r':
                    newS.append("\\r");
                    break;
                default:
                    newS.append(c);
                    break;
            }
        }
        return newS.toString();
    }

    private static class PolyEvalType {
        public String typeStr;
        public String typeName;
        public PolyEvalType valueType;
        public PolyEvalType keyType;

        public PolyEvalType(String typeStr) {
            this.typeStr = typeStr;
            if (!typeStr.contains("<")) {
                this.typeName = typeStr;
                return;
            }
            else {
                int idx = typeStr.indexOf("<");
                this.typeName = typeStr.substring(0, idx);
                String otherStr = typeStr.substring(idx + 1, typeStr.length() - 1);
                if (!otherStr.contains(",")) {
                    this.valueType = new PolyEvalType(otherStr);
                }
                else {
                    idx = otherStr.indexOf(",");
                    this.keyType = new PolyEvalType(otherStr.substring(0, idx));
                    this.valueType = new PolyEvalType(otherStr.substring(idx + 1));
                }
            }
        }
    }

    private static String genVoid(Object value) {
        assert value == null;
        return "null";
    }

    private static String genInt(Object value) {
        // ORIGINAL: assert value instanceof Integer;
        // ORIGINAL: return value.toString();
        
        // MODIFIED: Handle null values
        if (value == null) {
            return "null";
        }
        assert value instanceof Integer : "Expected Integer, got: " + value.getClass();
        return value.toString();
    }

    private static String genLong(Object value) {
        // ORIGINAL: assert value instanceof Long;
        // ORIGINAL: return value.toString() + "L";
        
        // MODIFIED: Handle null values
        if (value == null) {
            return "null";
        }
        assert value instanceof Long : "Expected Long, got: " + value.getClass();
        return value.toString() + "L";
    }

    private static String genDouble(Object value) {
        // ORIGINAL: assert value instanceof Double;
        // ORIGINAL: double f = (double) value;
        
        // MODIFIED: Handle null values
        if (value == null) {
            return "null";
        }
        assert value instanceof Double : "Expected Double, got: " + value.getClass();
        double f = (double) value;
        if (Double.isNaN(f)) {
            return "nan";
        }
        else if (Double.isInfinite(f)) {
            if (f > 0) {
                return "inf";
            }
            else {
                return "-inf";
            }
        }
        String valueStr = String.format("%.6f", f).replaceAll("0*$", "");
        if (valueStr.endsWith(".")) {
            valueStr += "0";
        }
        if (valueStr.equals("-0.0")) {
            valueStr = "0.0";
        }
        return valueStr;
    }

    private static String genBool(Object value) {
        // ORIGINAL: assert value instanceof Boolean;
        // ORIGINAL: return (Boolean) value ? "true" : "false";
        
        // MODIFIED: Handle null values
        if (value == null) {
            return "null";
        }
        assert value instanceof Boolean : "Expected Boolean, got: " + value.getClass();
        return (Boolean) value ? "true" : "false";
    }

    private static String genChar(Object value) {
        // ORIGINAL: assert value instanceof Character;
        // ORIGINAL: return "'" + escapeString(Character.toString((Character)value)) + "'";
        
        // MODIFIED: Handle null values
        if (value == null) {
            return "null";
        }
        assert value instanceof Character : "Expected Character, got: " + value.getClass();
        return "'" + escapeString(Character.toString((Character)value)) + "'";
    }

    private static String genString(Object value) {
        // ORIGINAL: assert value instanceof String;
        // ORIGINAL: return "\"" + escapeString((String) value) + "\"";
        
        // MODIFIED: Handle null values
        if (value == null) {
            return "null";
        }
        assert value instanceof String : "Expected String, got: " + value.getClass();
        return "\"" + escapeString((String) value) + "\"";
    }

    private static String genAny(Object value) {
        if (value instanceof Boolean) {
            return genBool(value);
        }
        else if (value instanceof Integer) {
            return genInt(value);
        }
        else if (value instanceof Long) {
            return genLong(value);
        }
        else if (value instanceof Double) {
            return genDouble(value);
        }
        else if (value instanceof Character) {
            return genChar(value);
        }
        else if (value instanceof String) {
            return genString(value);
        }
        assert false;
        return null;
    }

    private static String genList(Object value, PolyEvalType t) {
        // ORIGINAL: assert value instanceof List;
        // ORIGINAL: List<Object> list = (List<Object>) value;
        
        // MODIFIED: Handle both arrays and Lists to fix ClassCastException
        List<Object> list;
        if (value instanceof List) {
            list = (List<Object>) value;
        } else if (value instanceof int[]) {
            int[] array = (int[]) value;
            list = new ArrayList<>();
            for (int item : array) {
                list.add(item);
            }
        } else if (value instanceof double[]) {
            double[] array = (double[]) value;
            list = new ArrayList<>();
            for (double item : array) {
                list.add(item);
            }
        } else if (value instanceof Object[]) {
            Object[] array = (Object[]) value;
            list = Arrays.asList(array);
        } else {
            // Fallback: try to treat as List (original behavior)
            assert value instanceof List : "Expected List or Array, got: " + value.getClass();
            list = (List<Object>) value;
        }
        
        List<String> vStrs = new ArrayList<>();
        for (Object v : list) {
            vStrs.add(toPolyEvalStrWithType(v, t.valueType));
        }
        String vStr = String.join(", ", vStrs);
        return "[" + vStr + "]";
    }

    private static String genMlist(Object value, PolyEvalType t) {
        // ORIGINAL: assert value instanceof List;
        // ORIGINAL: List<Object> list = (List<Object>) value;
        
        // MODIFIED: Handle both arrays and Lists to fix ClassCastException
        List<Object> list;
        if (value instanceof List) {
            list = (List<Object>) value;
        } else if (value instanceof int[]) {
            int[] array = (int[]) value;
            list = new ArrayList<>();
            for (int item : array) {
                list.add(item);
            }
        } else if (value instanceof double[]) {
            double[] array = (double[]) value;
            list = new ArrayList<>();
            for (double item : array) {
                list.add(item);
            }
        } else if (value instanceof Object[]) {
            Object[] array = (Object[]) value;
            list = Arrays.asList(array);
        } else {
            // Fallback: try to treat as List (original behavior)
            assert value instanceof List : "Expected List or Array, got: " + value.getClass();
            list = (List<Object>) value;
        }
        
        List<String> vStrs = new ArrayList<>();
        for (Object v : list) {
            vStrs.add(toPolyEvalStrWithType(v, t.valueType));
        }
        String vStr = String.join(", ", vStrs);
        return "[" + vStr + "]";
    }

    private static String genUnorderedlist(Object value, PolyEvalType t) {
        // ORIGINAL: assert value instanceof List;
        // ORIGINAL: List<Object> list = (List<Object>) value;
        
        // MODIFIED: Handle both arrays and Lists to fix ClassCastException
        List<Object> list;
        if (value instanceof List) {
            list = (List<Object>) value;
        } else if (value instanceof int[]) {
            int[] array = (int[]) value;
            list = new ArrayList<>();
            for (int item : array) {
                list.add(item);
            }
        } else if (value instanceof double[]) {
            double[] array = (double[]) value;
            list = new ArrayList<>();
            for (double item : array) {
                list.add(item);
            }
        } else if (value instanceof Object[]) {
            Object[] array = (Object[]) value;
            list = Arrays.asList(array);
        } else {
            // Fallback: try to treat as List (original behavior)
            assert value instanceof List : "Expected List or Array, got: " + value.getClass();
            list = (List<Object>) value;
        }
        
        List<String> vStrs = new ArrayList<>();
        for (Object v : list) {
            vStrs.add(toPolyEvalStrWithType(v, t.valueType));
        }
        Collections.sort(vStrs);
        String vStr = String.join(", ", vStrs);
        return "[" + vStr + "]";
    }

    private static String genDict(Object value, PolyEvalType t) {
        assert value instanceof Map;
        Map<Object, Object> map = (Map<Object, Object>) value;
        List<String> vStrs = new ArrayList<>();
        for (Map.Entry<Object, Object> entry : map.entrySet()) {
            Object key = entry.getKey();
            Object val = entry.getValue();
            String kStr = toPolyEvalStrWithType(key, t.keyType);
            String vStr = toPolyEvalStrWithType(val, t.valueType);
            vStrs.add(kStr + "=>" + vStr);
        }
        Collections.sort(vStrs);
        String vStr = String.join(", ", vStrs);
        return "{" + vStr + "}";
    }

    private static String genMdict(Object value, PolyEvalType t) {
        assert value instanceof Map;
        Map<Object, Object> map = (Map<Object, Object>) value;
        List<String> vStrs = new ArrayList<>();
        for (Map.Entry<Object, Object> entry : map.entrySet()) {
            Object key = entry.getKey();
            Object val = entry.getValue();
            String kStr = toPolyEvalStrWithType(key, t.keyType);
            String vStr = toPolyEvalStrWithType(val, t.valueType);
            vStrs.add(kStr + "=>" + vStr);
        }
        Collections.sort(vStrs);
        String vStr = String.join(", ", vStrs);
        return "{" + vStr + "}";
    }

    private static String genOptional(Object value, PolyEvalType t) {
        // ORIGINAL: assert value instanceof Optional;
        // ORIGINAL: Optional<Object> optValue = (Optional<Object>) value;
        
        // MODIFIED: Handle both Optional objects and values that should be wrapped in Optional
        if (value == null) {
            return "null";
        }
        
        Optional<Object> optValue;
        if (value instanceof Optional) {
            // Already an Optional
            optValue = (Optional<Object>) value;
        } else {
            // Value that should be treated as Optional.of(value)
            // This handles cases where model returns String but framework expects Optional<String>
            optValue = Optional.of(value);
        }
        
        if (optValue.isPresent()) {
            return toPolyEvalStr(optValue.get(), t.valueType);
        }
        else {
            return "null";
        }
    }

    private static String toPolyEvalStr(Object value, PolyEvalType t) {
        String typeName = t.typeName;
        if (typeName.equals("void")) {
            return genVoid(value);
        }
        else if (typeName.equals("int")) {
            return genInt(value);
        }
        else if (typeName.equals("long")) {
            return genLong(value);
        }
        else if (typeName.equals("double")) {
            return genDouble(value);
        }
        else if (typeName.equals("bool")) {
            return genBool(value);
        }
        else if (typeName.equals("char")) {
            return genChar(value);
        }
        else if (typeName.equals("string")) {
            return genString(value);
        }
        else if (typeName.equals("any")) {
            return genAny(value);
        }
        else if (typeName.equals("list")) {
            return genList(value, t);
        }
        else if (typeName.equals("mlist")) {
            return genMlist(value, t);
        }
        else if (typeName.equals("unorderedlist")) {
            return genUnorderedlist(value, t);
        }
        else if (typeName.equals("dict")) {
            return genDict(value, t);
        }
        else if (typeName.equals("mdict")) {
            return genMdict(value, t);
        }
        else if (typeName.equals("optional")) {
            return genOptional(value, t);
        }
        assert false;
        return null;
    }

    private static String toPolyEvalStrWithType(Object value, PolyEvalType t) {
        return toPolyEvalStr(value, t) + ":" + t.typeStr;
    }

    public static String myStringify(Object value, String typeStr) {
        return toPolyEvalStrWithType(value, new PolyEvalType(typeStr));
    }
}

$$code$$