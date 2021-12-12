import React from "react";
import styled from "styled-components";

const COLORS = [
  "Crimson",
  "DarkGreen",
  "DodgerBlue",
  "MediumPurple",
  "DeepPink",
  "Gold",
  "LimeGreen",
  "DarkOrange",
] as const;
const BaseMemoryCard = styled.div`
  position: absolute;
  top: 0;
  bottom: 0;
  left: 0;
  right: 0;
  display: flex;
  flex: 1 0 21%;
  border-radius: 0.5rem;
  align-items: center;
  justify-content: center;
  backface-visibility: hidden;
  &::before {
    border-radius: 0.25rem;
  }
`;
const RevealedMemoryCard = styled(BaseMemoryCard)<{ readonly cardId: number }>`
  background-color: ${({ cardId }) => COLORS[cardId]};
  transform: rotateY(180deg);
  &::before {
    content: "${({ cardId }) => COLORS[cardId]}";
    background-color: white;
    padding: 1rem;
  }
`;
const HiddenMemoryCard = styled(BaseMemoryCard)<{ readonly cardId: number }>`
  background-color: white;
  border: medium solid black;
  padding: 0.25rem;
  &::before {
    content: "";
    flex: 1;
    background-color: SaddleBrown;
  }
`;
const CardWrapper = styled.div`
  perspective: min(100vh, 100vw);
  flex: 1 0 21%;
  aspect-ratio: 1;
`;
const CardFlipper = styled.div<{ readonly isFlipped: boolean }>`
  height: 100%;
  width: 100%;
  position: relative;
  transition: transform 1s;
  transform-style: preserve-3d;
  box-shadow: rgba(0, 0, 0, 0.24) 0px 4px 8px;
  border-radius: 0.5rem;
  transform: rotateY(${({ isFlipped }) => (isFlipped ? 180 : 0)}deg);
`;

interface CardProps {
  readonly cardId: number;
  readonly revealed: boolean;
  readonly onClick: () => void;
}

export const Card: React.VFC<CardProps> = ({ cardId, revealed, onClick }) => {
  return (
    <CardWrapper onClick={onClick}>
      <CardFlipper isFlipped={revealed}>
        <RevealedMemoryCard cardId={cardId} />
        <HiddenMemoryCard cardId={cardId} />
      </CardFlipper>
    </CardWrapper>
  );
};
