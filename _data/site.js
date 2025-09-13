module.exports = {
  meta: {
    title: "Prathyush Maniyam",
    description: "Lorem ipsum dolor sit amet consectetur adipisicing elit.",
    lang: "en",
    siteUrl: "https://example.com/",
  },
  feed: { // used in feed.xml.njk
    subtitle: "Lorem ipsum dolor sit amet consecuteor",
    filename: "atom.xml",
    path: "/atom.xml",
    id: "https://example.com/",
    authorName: "John Doe",
    authorEmail: "johndoe@example.com"
  },
  hero: { // used in hero section of main page ie. index.html.njk
    title: "Welcome to my Portfolio",
    description: "Hey, I’m pratzzeee 👋 I’m a 29-year-old grad student who finished a master’s in Business Analytics at UTD. Lately, I’ve been obsessed with learning new skills, pushing my limits, and hacking my brain."
  }
}