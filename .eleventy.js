const path = require("path");
const sass = require("sass");

module.exports = function (eleventyConfig) {
  // Passthrough copy for static assets
  eleventyConfig.addPassthroughCopy("assets");
  // If you have other static directories (like images directly in root), add them here
  // eleventyConfig.addPassthroughCopy("images");

  // Tell Eleventy to ignore build/tooling/source control files
  eleventyConfig.ignores.add("node_modules/**");
  eleventyConfig.ignores.add(".git/**");
  eleventyConfig.ignores.add(".trunk/**");
  eleventyConfig.ignores.add("_site/**"); // Ignore the output directory
  eleventyConfig.ignores.add("Gemfile");
  eleventyConfig.ignores.add("Gemfile.lock");
  eleventyConfig.ignores.add("README.md"); // Usually not part of the site
  // eleventyConfig.ignores.add("_config.yml"); // Ignore Jekyll config later

  // Explicitly tell the watcher to ignore .trunk directory changes
  eleventyConfig.watchIgnores.add(".trunk/**");

  // Add the year shortcode
  eleventyConfig.addShortcode("year", () => `${new Date().getFullYear()}`);

  // Add Sass support
  eleventyConfig.addTemplateFormats("scss");
  eleventyConfig.addExtension("scss", {
    outputFileExtension: "css",
    compileOptions: {
      permalink: function (contents, inputPath) {
        // Ensure files starting with _ are not directly output
        let parsed = path.parse(inputPath);
        if (parsed.name.startsWith("_")) {
          return false; // Don't output this file
        }
        // For all other Sass files, return undefined to use default permalink behavior
        return undefined;
      },
    },
    compile: async function (inputContent, inputPath) {
      // Skip files starting with _
      let parsed = path.parse(inputPath);
      if (parsed.name.startsWith("_")) {
        return; // Do not compile this file
      }

      // Find the main entry point for Sass compilation
      // Assuming your main Sass file is `_sass/main.scss`
      // You might need to adjust this if your main entry point is different
      let mainSassPath = path.join(path.dirname(inputPath), "main.scss");
      if (
        inputPath !== mainSassPath &&
        !inputContent.includes("@import") &&
        !inputContent.includes("@use")
      ) {
        // If it's not the main file and doesn't seem to import anything,
        // assume it's a partial meant to be included elsewhere and skip direct compilation.
        // A more robust approach might involve checking dependency graphs,
        // but this handles simple cases.
        // Alternatively, if you *only* want `_sass/main.scss` compiled,
        // check `if(inputPath === mainSassPath)` here.
        // For now, we'll compile *any* non-partial file.
      }

      let result = sass.compileString(inputContent, {
        // Tell Sass where to look for @import/@use statements
        loadPaths: [parsed.dir || ".", "_sass"],
        // Optional: Use 'compressed' for production builds
        // style: "compressed"
      });

      // This is the render function, `data` is the full data cascade
      return async (data) => {
        return result.css;
      };
    },
  });

  // Configuration for Liquid templating (if we keep using it)
  eleventyConfig.setLiquidOptions({
    dynamicPartials: false, // Recommended for consistency
    strictFilters: false, // Might be needed for Jekyll compatibility
  });

  // Add other Eleventy plugins or configurations here later (e.g., sitemap)

  return {
    // Control which files Eleventy will process
    templateFormats: [
      "md",
      "liquid", // Process Jekyll's .html layouts/includes if they use Liquid
      "njk", // Allow Nunjucks templates too
      "html", // Process plain HTML files
      // Add other formats like "11ty.js" if needed
    ],

    // Pre-process Markdown files with Liquid (for Liquid tags inside Markdown)
    markdownTemplateEngine: "njk",

    // Pre-process HTML files with Liquid (for Liquid tags inside HTML)
    htmlTemplateEngine: "liquid",

    dir: {
      input: ".", // Root directory for sources
      includes: "_includes", // Default includes directory
      layouts: "_layouts", // Default layouts directory
      data: "_data", // Default data directory
      output: "_site", // Default output directory
    },
  };
};
