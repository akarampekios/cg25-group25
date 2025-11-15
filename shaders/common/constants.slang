public static const float PI = 3.14159265359;

// Acceleration structure instance masks (bit flags)
public static const uint AS_REFLECTIVE_MASK = 0x01; // Object can be hit by reflection rays
public static const uint AS_SHADOW_MASK = 0x02;     // Object can cast shadows
public static const uint AS_LIT_OBJECT_MASK = 0x01; // Alias for reflective mask for backward compatibility
public static const uint AS_SHADOW_OBJECT_MASK = 0x02; // Alias for shadow mask
public static const uint AS_UNKNOWN_OBJ_MASK = 0x00; // Object is invisible to ray tracing

public static const uint MAX_TEXTURE_ARRAY_SIZE = 1024;
