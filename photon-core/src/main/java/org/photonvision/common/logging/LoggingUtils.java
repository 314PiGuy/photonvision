package org.photonvision.common.logging;

/**
 * Small helper to provide static logger instances for classes that need a logger in static contexts
 * or helper methods.
 */
public class LoggingUtils {
    /** Get a logger for the given class. */
    public static Logger getLogger(Class<?> cls, String context) {
        return new Logger(cls, context, LogGroup.General);
    }
}
