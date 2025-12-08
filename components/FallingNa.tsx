import React from 'react';
import { FallingNa as FallingNaType } from '../types';

interface FallingNaProps {
  na: FallingNaType;
}

const FallingNa: React.FC<FallingNaProps> = ({ na }) => {
  return (
    <span
      key={na.id}
      className="absolute top-0 text-white font-black drop-shadow-[0_4px_0_rgba(236,72,153,0.5)] animate-fall z-0 pointer-events-none select-none"
      style={{
        left: na.left,
        fontSize: na.size,
        animationDuration: na.duration,
        WebkitTextStroke: '3px #f472b6', // pink-400
      }}
    >
      な
    </span>
  );
};

export default FallingNa;