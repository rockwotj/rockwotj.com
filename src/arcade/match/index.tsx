import React, {useState} from "react";
import styled from "styled-components";
import { useMemoryMatchGame } from "./logic";
import { useConfetti } from "../../hooks/useConfetti";
import { GameOverBanner, GameOverOverlay } from "./GameOver";
import { Card } from "./Card";
import {useTimeout} from "../../hooks/useTimeout";

const CardGrid = styled.div`
  position: relative; // For the overlay banner
  display: flex;
  align-items: flex-start;
  justify-content: flex-start;
  flex-wrap: wrap;
  gap: 0.5rem;
  max-width: min(100vh, 100vw);
  max-height: min(100vh, 100vw);
  padding: 1rem;
  margin: 0 auto;
`;

export const MemoryMatchGame: React.VFC = () => {
  const { cards, revealed, onSelected, gameOver } = useMemoryMatchGame();
  const [showGameOver, setShowGameOver] = useState(false);
  useTimeout(() => setShowGameOver(true), gameOver ? 1000 : null);
  useConfetti({ makeItRain: showGameOver });
  return (
    <CardGrid>
      {cards.map((cardId, idx) => (
        <Card
          key={idx}
          cardId={cardId}
          revealed={revealed.has(idx)}
          onClick={() => onSelected(idx)}
        />
      ))}
      {showGameOver ? (
        <GameOverOverlay>
          <GameOverBanner />
        </GameOverOverlay>
      ) : null}
    </CardGrid>
  );
};
