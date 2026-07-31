import { useState } from "react";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");
  const [isSearching, setIsSearching] = useState(false);

  const visibleResults = results.slice(0, 20);

  async function search() {
    if (!query.trim()) return;

    setIsSearching(true);
    setError("");

    try {
      const response = await fetch(
        `http://localhost:8000/search?q=${encodeURIComponent(query)}`
      );

      if (!response.ok) {
        throw new Error("Request failed");
      }

      const data = await response.json();
      setResults(data.results || []);
    } catch (err) {
      setError("Unable to reach the backend. Make sure the FastAPI server is running.");
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="background-glow glow-one" />
      <div className="background-glow glow-two" />

      <main className="container">
        <section className="hero-panel">
          <div className="hero-copy">
            <p className="eyebrow">Curated discovery</p>
            <h1>Find what fits your taste.</h1>
            <p className="subtext">
              A calm, premium way to explore recommendations from your catalog.
            </p>
          </div>

          <div className="search-card">
            <div className="search-box">
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search for artists, tracks, or moods..."
                onKeyDown={(e) => e.key === "Enter" && search()}
              />
              <button onClick={search} disabled={isSearching}>
                {isSearching ? "Searching..." : "Search"}
              </button>
            </div>

            {error ? <p className="error-text">{error}</p> : null}
          </div>

          <div className="results-panel">
            {results.length > 0 ? (
              <>
                <div className="results-header">
                  <p>Showing top 20 recommendations</p>
                </div>
                <div className="results-grid">
                  {visibleResults.map((item, index) => (
                    <article className="card" key={`${item.id || item.title}-${index}`}>
                      <div className="card-topline">
                        <span className="play-button" aria-label="Play" />
                        <span className="score-badge">{Number(item.score ?? 0).toFixed(2)}</span>
                      </div>
                      <h3>{item.title}</h3>
                      <p className="card-meta">Artist • Track • Mood</p>
                      <div className="card-footer">
                        <span>Recommended for you</span>
                      </div>
                    </article>
                  ))}
                </div>
              </>
            ) : (
              <div className="empty-state">
                <p>Start with a query to reveal refined recommendations.</p>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;