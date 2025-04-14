module.exports = function (eleventyConfig) {
  // Tell Eleventy to ignore specific directories/files
  eleventyConfig.ignores.add("node_modules/**");
  eleventyConfig.ignores.add(".git/**");
  eleventyConfig.ignores.add(".trunk/**");
  // Add any other directories you want Eleventy to completely ignore

  // Passthrough copy - static assets from 'assets' directory
  eleventyConfig.addPassthroughCopy("assets");

  // Return your Object options:
  return {
    dir: {
      input: ".", // Source files directory (current directory for now)
      output: "_site", // Build output directory
      includes: "_includes", // Relative to input directory
      layouts: "_layouts", // Relative to input directory
      data: "_data", // Relative to input directory
    },
    // Files read by Eleventy, add as needed
    templateFormats: [
      "md",
      "njk", // Nunjucks templates
      "html",
      "liquid", // Liquid templates (like Jekyll)
    ],
    // Passthrough copy setting (can often be omitted when using addPassthroughCopy)
    // passthroughFileCopy: true // We are using addPassthroughCopy above, so this might be redundant but doesn't hurt
  };
};

// We'll add the passthrough for your 'assets' directory later.
