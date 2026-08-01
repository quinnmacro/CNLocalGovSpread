"use client";

import { useEffect, useState } from "react";

/**
 * IntersectionObserver-based scroll spy for TOC navigation.
 * Returns the id of the currently visible section.
 * @param ready - pass false while loading so observer registers after DOM elements exist
 */
export function useScrollSpy(
  sectionIds: readonly string[],
  offset = 120,
  ready = true,
): string {
  const [activeId, setActiveId] = useState<string>(sectionIds[0] ?? "");

  useEffect(() => {
    if (!ready) return;

    let observer: IntersectionObserver | null = null;

    // Small delay to let React render sections into DOM
    const timer = setTimeout(() => {
      observer = new IntersectionObserver(
        (entries) => {
          const visible = entries
            .filter((e) => e.isIntersecting)
            .sort(
              (a, b) => a.boundingClientRect.top - b.boundingClientRect.top,
            );
          if (visible.length > 0) {
            setActiveId(visible[0].target.id);
          }
        },
        {
          rootMargin: `-${offset}px 0px -40% 0px`,
          threshold: [0, 0.1, 0.25],
        },
      );

      const elements = sectionIds
        .map((id) => document.getElementById(id))
        .filter((el): el is HTMLElement => el != null);

      for (const el of elements) {
        observer.observe(el);
      }
    }, 50);

    return () => {
      clearTimeout(timer);
      observer?.disconnect();
    };
  }, [sectionIds, offset, ready]);

  return activeId;
}
