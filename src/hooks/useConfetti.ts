import confetti from "canvas-confetti";

import { isNil, isNumber, random } from "lodash-es";
import { useEffect, useRef } from "react";

interface UseConfettiProps {
  readonly makeItRain: boolean;
  readonly duration?: number;
}

const CONFETTI_OPTIONS: confetti.Options = {
  startVelocity: 30,
  spread: 360,
  ticks: 60,
  zIndex: 0,
  disableForReducedMotion: true,
};
const DEFAULT_DURATION = 10 * 1000;

export function useConfetti({
  makeItRain,
  duration = DEFAULT_DURATION,
}: UseConfettiProps) {
  const intervalHandle = useRef<ReturnType<typeof setInterval> | null>(null);
  let animationEnd = Date.now() + duration;

  useEffect(() => {
    return () => {
      if (isNumber(intervalHandle.current)) {
        clearInterval(intervalHandle.current);
      }
      confetti.reset();
    };
  });

  if (makeItRain === isNumber(intervalHandle.current)) return;

  if (isNil(intervalHandle.current)) {
    let myHandle = setInterval(function () {
      const timeLeft = animationEnd - Date.now();

      if (timeLeft <= 0) {
        return clearInterval(myHandle);
      }

      const particleCount = 50 * (timeLeft / duration);
      // since particles fall down, start a bit higher than random
      confetti({
        ...CONFETTI_OPTIONS,
        particleCount,
        origin: { x: random(0.1, 0.3), y: Math.random() - 0.2 },
      });
      confetti({
        ...CONFETTI_OPTIONS,
        particleCount,
        origin: { x: random(0.7, 0.9), y: Math.random() - 0.2 },
      });
    }, 250);
    intervalHandle.current = myHandle;
  } else {
    clearInterval(intervalHandle.current);
    intervalHandle.current = null;
  }
}
