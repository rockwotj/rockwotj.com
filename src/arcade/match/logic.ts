import { useSet } from "react-use";
import { useCallback, useMemo } from "react";
import { concat, range, shuffle } from "lodash-es";
import { useTimeout } from "../../hooks/useTimeout";

export const NUM_PAIRS = 8;

/**
 * Create a 4x4 memory match game.
 */
export function useMemoryMatchGame() {
  const cards = useMemo(() => {
    return shuffle(concat(range(0, NUM_PAIRS), range(0, NUM_PAIRS)));
  }, []);

  const [solved, { add: addSolved }] = useSet<number>();
  const [selected, { toggle: toggleSelected, reset: resetSelected }] =
    useSet<number>();

  const onSelected = useCallback(
    (selection: number) => {
      if (selected.size > 1) return; // Wait for animation to finish.
      if (solved.has(selection)) return;
      if (selected.has(selection)) {
        return toggleSelected(selection);
      }
      const [otherSelection] = [...selected];
      if (cards[selection] === cards[otherSelection]) {
        addSolved(selection);
        addSolved(otherSelection);
        resetSelected();
      } else {
        toggleSelected(selection);
      }
    },
    [cards, selected, solved, toggleSelected, resetSelected]
  );

  useTimeout(
    () => {
      resetSelected();
    },
    selected.size > 1 ? 1000 : null
  );

  return {
    revealed: new Set([...solved, ...selected]),
    cards,
    onSelected,
    gameOver: NUM_PAIRS * 2 === solved.size,
  };
}
