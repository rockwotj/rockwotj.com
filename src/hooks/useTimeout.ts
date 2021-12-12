import { useEffect, useRef } from "react";

/**
 * A wrapper around setTimeout that handles delay and callback changes.
 *
 * @param callback The callback that fires upon the timeout expiring.
 * @param delay The delay before the timeout expires. Use null to disable the timeout.
 */
export function useTimeout(callback: () => void, delay: number | null) {
  const savedCallback = useRef(callback);
  savedCallback.current = callback;

  // Set up the timeout.
  useEffect(() => {
    // Don't schedule if no delay is specified.
    if (delay === null) return;

    const timeoutId = setTimeout(() => savedCallback.current(), delay);

    return () => clearTimeout(timeoutId);
  }, [delay]);
}
