import { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { type Artwork, asset, loadGallery, readFavorites, readableDate } from './gallery'
import './style.css'

function useHash() {
  const [hash, setHash] = useState(window.location.hash)
  useEffect(() => {
    const listen = () => setHash(window.location.hash)
    window.addEventListener('hashchange', listen)
    return () => window.removeEventListener('hashchange', listen)
  }, [])
  return hash
}

function Mark() {
  return <span className="brand-mark" aria-hidden="true"><i/><i/><i/></span>
}

function FavoriteButton({ active, toggle }: { active: boolean; toggle: () => void }) {
  return <button type="button" className="favorite" aria-pressed={active}
    aria-label={active ? 'Remove from favorites' : 'Add to favorites'} onClick={toggle}>{active ? '♥' : '♡'}</button>
}

function Tile({ work, active, toggle }: { work: Artwork; active: boolean; toggle: () => void }) {
  return <article className="tile">
    <div className="tile-image">
      <a href={`#/art/${work.id}`} aria-label={`View ${work.title}`}>
        <img src={asset(work.thumb)} alt={work.title} width={work.width} height={work.height} loading="lazy"/>
      </a>
      <FavoriteButton active={active} toggle={toggle}/>
    </div>
    <a className="tile-caption" href={`#/art/${work.id}`}>
      <h3>{work.title}</h3>
      <span>{work.theme}</span>
    </a>
  </article>
}

function Detail({ artwork, active, toggle, back }: {
  artwork: Artwork; active: boolean; toggle: () => void; back: string
}) {
  useEffect(() => {
    window.scrollTo(0, 0)
    document.getElementById('art-title')?.focus()
  }, [artwork.id])

  return <main className="shell detail">
    <a className="back" href={back}>Back to collection</a>
    <div className="detail-grid">
      <a className="original" href={asset(artwork.image)} target="_blank" rel="noreferrer"
        aria-label={`View full resolution of ${artwork.title}`}>
        <img src={asset(artwork.image)} alt={artwork.title} width={artwork.width} height={artwork.height}/>
      </a>
      <section className="detail-meta">
        <p className="eyebrow">{artwork.theme}</p>
        <h1 tabIndex={-1} id="art-title">{artwork.title}</h1>
        <p className="muted">{readableDate(artwork.date)} · {artwork.width} × {artwork.height}</p>
        <div className="detail-actions">
          <a className="button accent" href={asset(artwork.image)} download={`${artwork.id}.png`}>Download image</a>
          <FavoriteButton active={active} toggle={toggle}/>
        </div>
        <dl>
          <div><dt>Model</dt><dd>{artwork.model}</dd></div>
          <div><dt>Seed</dt><dd>{artwork.seed}</dd></div>
          <div><dt>Steps</dt><dd>{artwork.steps}</dd></div>
          <div><dt>Palette</dt><dd>{artwork.palette}</dd></div>
          <div><dt>Composition</dt><dd>{artwork.composition}</dd></div>
        </dl>
        <h2>Prompt</h2>
        <p className="prompt">{artwork.prompt}</p>
      </section>
    </div>
  </main>
}

function App() {
  const [works, setWorks] = useState<Artwork[] | null>(null)
  const [error, setError] = useState(false)
  const [retry, setRetry] = useState(0)
  const [favorites, setFavorites] = useState(readFavorites)
  const [theme, setTheme] = useState('')
  const [month, setMonth] = useState('')
  const [previous, setPrevious] = useState('#collection')
  const hash = useHash()
  const favoritesOnly = hash === '#favorites'
  const detailId = hash.startsWith('#/art/') ? hash.slice(6) : null
  const selected = detailId ? works?.find((work) => work.id === detailId) : null

  useEffect(() => {
    const ctrl = new AbortController()
    loadGallery(ctrl.signal).then(setWorks).catch(() => {
      if (!ctrl.signal.aborted) setError(true)
    })
    return () => ctrl.abort()
  }, [retry])

  useEffect(() => {
    if (!detailId) setPrevious(favoritesOnly ? '#favorites' : '#collection')
    if (hash === '#collection' || favoritesOnly || hash === '#monthly') {
      requestAnimationFrame(() => document.getElementById(hash.slice(1))?.scrollIntoView())
    }
  }, [hash, detailId, favoritesOnly, works])

  const toggle = (id: string) => {
    const updated = favorites.includes(id) ? favorites.filter((item) => item !== id) : [...favorites, id]
    setFavorites(updated)
    try { localStorage.setItem('museum-favorites-v1', JSON.stringify(updated)) } catch { /* Session state remains available. */ }
  }
  const themes = [...new Set(works?.map((work) => work.theme))].sort()
  const dates = [...new Set(works?.map((work) => work.date.slice(0, 7)))].sort().reverse()
  const visible = works?.filter((work) => (!favoritesOnly || favorites.includes(work.id))
    && (!theme || work.theme === theme) && (!month || work.date.startsWith(month)))
  const featured = works?.filter((work) => work.date.startsWith(month || dates[0]))
    .sort((a, b) => Number(favorites.includes(b.id)) - Number(favorites.includes(a.id))
      || b.approved_at.localeCompare(a.approved_at)).slice(0, 3)
  const card = (work: Artwork) => <Tile key={work.id} work={work} active={favorites.includes(work.id)}
    toggle={() => toggle(work.id)}/>
  const unavailable = error ? <div role="alert" className="empty">
    <h3>The gallery is temporarily unavailable.</h3>
    <button className="button" onClick={() => { setError(false); setRetry(retry + 1) }}>Try again</button>
  </div> : <p className="empty muted" role="status">Loading the collection…</p>

  return <>
    <header className="header">
      <div className="shell header-inner">
        <a className="brand" href="#" aria-label="AI Art Museum home"><Mark/>AI Art Museum</a>
        <nav aria-label="Main navigation">
          <a href="#collection" aria-current={!favoritesOnly && !detailId ? 'location' : undefined}>Collection</a>
          <a href="#monthly">Monthly picks</a>
          <a href="#favorites" aria-current={favoritesOnly ? 'location' : undefined}><span aria-hidden="true">♡ </span>Favorites</a>
        </nav>
      </div>
    </header>
    {detailId ? (selected ? <Detail artwork={selected} active={favorites.includes(selected.id)}
      toggle={() => toggle(selected.id)} back={previous}/> : <main className="shell missing">
      {works ? <><h1>Artwork unavailable</h1><a href="#collection">Back to collection</a></> : unavailable}
    </main>) : <main>
      {!favoritesOnly && <section className="hero">
        <div className="hero-art" aria-hidden="true"><div className="halo"/><div className="orbital orbital-a"/>
          <div className="orbital orbital-b"/><div className="orbital orbital-c"/><div className="orbital orbital-d"/></div>
        <div className="shell hero-content">
          <p className="eyebrow">An evolving collection</p>
          <h1>AI Art <em>Museum</em></h1>
          <p>Curated by Human, Created by AI</p>
          <a className="button accent" href="#collection">Explore the collection</a>
        </div>
      </section>}
      <section className="shell collection" id={favoritesOnly ? 'favorites' : 'collection'}>
        <div className="section-heading">
          <div>
            <p className="eyebrow">{favoritesOnly ? 'Saved on this device' : 'The archive'}</p>
            <h2>{favoritesOnly ? 'Favorites' : 'Collection'}</h2>
          </div>
          <div className="filters">
            <label><span>Theme</span><select value={theme} onChange={(event) => setTheme(event.target.value)}>
              <option value="">All themes</option>{themes.map((name) => <option key={name}>{name}</option>)}
            </select></label>
            <label><span>Date</span><select value={month} onChange={(event) => setMonth(event.target.value)}>
              <option value="">All dates</option>{dates.map((value) => <option value={value} key={value}>{readableDate(value, true)}</option>)}
            </select></label>
          </div>
        </div>
        {works?.some((work) => work.mock) && <p className="demo" role="status">Private demo · simulated artworks</p>}
        {visible ? visible.length ? <div className="masonry">{visible.map(card)}</div>
          : <div className="empty">
            <h3>{favoritesOnly && favorites.length === 0 ? 'Your favorites will appear here.'
              : works?.length ? 'No works match these filters.' : 'The collection opens with its first approved work.'}</h3>
            {(theme || month) && <button className="button" onClick={() => { setTheme(''); setMonth('') }}>Clear filters</button>}
            {favoritesOnly && favorites.length === 0 && <a href="#collection">Browse the collection</a>}
          </div> : unavailable}
      </section>
      {!favoritesOnly && <section className="shell monthly" id="monthly">
        <p className="eyebrow">{(month || dates[0]) ? readableDate(month || dates[0], true) : 'The archive'}</p>
        <h2>Monthly picks</h2>
        {featured?.length ? <div className="picks">{featured.map(card)}</div>
          : <p className="muted">Monthly picks appear as works enter the collection.</p>}
      </section>}
    </main>}
    <footer className="footer"><div className="shell"><a className="brand" href="#"><Mark/>AI Art Museum</a>
      <span>Curated by Human, Created by AI</span></div></footer>
  </>
}

createRoot(document.getElementById('root')!).render(<App/>)
