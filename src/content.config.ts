import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const aboutCollection = defineCollection({
  loader: glob({
    pattern: "**/*.{md,mdx}",
    base: "./src/content/about",
  }),
  schema: z.object({
    title: z.string(),
    lastUpdated: z.coerce.date().optional(),
  }),
});

export const collections = {
  about: aboutCollection,
};
