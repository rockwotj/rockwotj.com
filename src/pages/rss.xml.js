import rss from '@astrojs/rss';

export async function get(context) {
  const postImports = import.meta.glob('./blog/*.md', { eager: true });
  const posts = Object.values(postImports)
    .filter(post => post.frontmatter.title && !post.frontmatter.draft)
    .sort((a, b) => new Date(b.frontmatter.date) - new Date(a.frontmatter.date));

  return rss({
    title: "Tyler Rockwood's Blog",
    description: 'Technical writings about software engineering, databases, WebAssembly, and more.',
    site: context.site,
    items: posts.map(post => ({
      title: post.frontmatter.title,
      pubDate: new Date(post.frontmatter.date),
      link: post.url,
    })),
  });
}
