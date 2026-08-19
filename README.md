# spam detector

This project was for me to stop writing models from scratch and actually see what sklearn can do when pushed. I’ve always had a love-hate relationship with high-level frameworks. When you just instantiate a pipeline, call .fit() and let scikit-learn handle the heavy lifting, you lose sight of what’s happening under the hood—the math, the feature extraction, the probability updates and all the edge cases get swallowed by abstraction. Having spent time manually building machine learning algorithms and neural networks, moving to a single .fit() call felt almost suspiciously easy, like missing out on the fun part.

My usual habit is forcing myself through the most tedious implementation details first, so switching to a clean library setup felt backwards. But this time around, the goal wasn to build something fast, clean and resilient.

I picked SMS spam detection because short-text classification brings its own set of weird quirks. People mess up encodings, send empty messages or use odd characters that choke standard parsers. Instead of just dumping everything into a loose script, I wrapped the whole pipeline into a RobustSpamDetector class.

The core ML part is a standard combo: a sublinear TF-IDF vectorizer paired with a multinomial naive bayes classifier. I included bigrams (ngram_range=(1,2)) so the model wouldn't just look at isolated words, but also catch specific two-word combinations like "win cash" or "claim now." After setting up sublinear term frequency scaling and additive smoothing, the actual training took just a few lines. Coming from manual implementation, getting an out-of-the-box ~98% accuracy in a split second felt absurdly abstract. You just pass in the vectors, and the model hands back a probability.

Since I didn't want the code breaking on real-world edge cases, I spent most of my effort making the pipeline unbreakable rather than tweaking hyperparameters. A lot of online examples assume your input CSV is perfectly formatted UTF-8 but real datasets rarely are. I wrote an automated fallback system that scans through utf-8, latin-1, iso-8859-1 and cp1252 until the data loads without error. I also added handling for extra dirty columns, empty string payloads and bad data types during live predictions.

Instead of just spitting out a single accuracy metric, the predict() method returns structured JSON containing the predicted label, a boolean flag and a confidence score. Watching Naive Bayes assign crisp probability bounds to incoming messages—and staying rock solid even when handed total gibberish or empty inputs—was easily the best part of the build.

Keeping the whole project modular, fully error-checked and self-contained in under 200 lines turned out to be a really satisfying trade-off between low-level control and framework efficiency.
