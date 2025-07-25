import { defineCollection, z } from "astro:content";

const aboutCollection = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    lastUpdated: z.date().optional(),
  }),
});

export const collections = {
  about: aboutCollection,
};
