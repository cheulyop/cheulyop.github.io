export const siteConfig = {
  name: "Cheul",
  email: "cheulyop@gmail.com",
  description: "Software Engineer passionate about AI and data",
  url: "https://cheul.dev",

  // Social links
  social: {
    github: "cheulyop",
    linkedin: "cheulyop",
    orcid: "0000-0003-0414-272X",
    twitter: null,
    scholar: null,
    medium: null,
  },

  // Analytics
  googleAnalytics: "UA-163428563-1",

  // Metadata
  lastUpdated: new Date("2025-01-24"),
} as const;

// Type export for use in other files
export type SiteConfig = typeof siteConfig;
