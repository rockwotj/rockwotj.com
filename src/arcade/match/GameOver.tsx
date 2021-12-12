import styled from "styled-components";

export const GameOverOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
`;

export const GameOverBanner = styled.h1.attrs({ children: "You Won!" })`
  background-color: #fafafa;
  padding: 3rem;
  border-radius: 0.5rem;
  box-shadow: rgba(0, 0, 0, 0.24) 0px 8px 16px;
`;
