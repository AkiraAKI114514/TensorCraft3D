import { createContext, useContext, useState } from 'react';

/**
 * The 3D view's look. `night` is the cinematic default: a dark stage, blocks as dark glass whose
 * cell lines and edges glow in their own colour, luminous data flow and bloom. `day` is the light
 * studio look, which exported images always use, whatever the view shows.
 */
export type SceneTheme = 'night' | 'day';

const STORAGE_KEY = 'tensorlab-scene-theme';
function stored(): SceneTheme {
  try { return localStorage.getItem(STORAGE_KEY) === 'day' ? 'day' : 'night'; } catch { return 'night'; }
}

/** The chosen look, remembered per browser. */
export function useSceneTheme(): [SceneTheme, () => void] {
  const [theme, setTheme] = useState<SceneTheme>(stored);
  const toggle = () => {
    const next = theme === 'night' ? 'day' : 'night';
    setTheme(next);
    try { localStorage.setItem(STORAGE_KEY, next); } catch { /* not remembered */ }
  };
  return [theme, toggle];
}

export const SceneThemeContext = createContext<SceneTheme>('day');
export const useSceneLook = () => useContext(SceneThemeContext);
