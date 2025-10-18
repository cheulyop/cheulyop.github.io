import { defineCollection, z } from "astro:content";

const aboutCollection = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    lastUpdated: z.date().optional(),
  }),
});

const postsCollection = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.date(),
    updatedDate: z.date().optional(),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
    heroImage: z
      .object({
        src: z.string(),
        alt: z.string(),
      })
      .optional(),
    defaultLanguage: z.string().default("en"),
    languages: z
      .array(
        z.object({
          code: z.string(),
          label: z.string(),
        }),
      )
      .default([{ code: "en", label: "English" }]),
    translations: z.record(z.string(), z.string()).default({}),
  }),
});

export const collections = {
  about: aboutCollection,
  posts: postsCollection,
};
