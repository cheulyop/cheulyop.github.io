module.exports = function (eleventyConfig) {
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
    // Passthrough copy - static assets
    // Add assets directory to copy static files (CSS, images, etc.)
    passthroughFileCopy: true,
  };
};

// We'll add the passthrough for your 'assets' directory later.
