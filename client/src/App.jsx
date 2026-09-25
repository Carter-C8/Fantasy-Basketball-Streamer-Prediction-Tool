import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

const POSITIONS = ['', 'Guard', 'Forward', 'Center']

function App() {
  const navigate = useNavigate()
  const [streamers, setStreamers] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const [weekOffset, setWeekOffset] = useState(0) // 0 = This Week, 1 = Next Week

  // NEW: whether the ESPN filter form is expanded or collapsed
  const [showFilterForm, setShowFilterForm] = useState(false)

  // NEW: position filter ('' means "All") and the historical top-156 toggle
  const [position, setPosition] = useState('')
  const [hideTop156, setHideTop156] = useState(false)

  // State for the ESPN Credentials
  const [leagueId, setLeagueId] = useState(localStorage.getItem('espn_league_id') || '')
  const [swid, setSwid] = useState(localStorage.getItem('espn_swid') || '')
  const [s2, setS2] = useState(localStorage.getItem('espn_s2') || '')

  const fetchStreamers = async () => {
    setLoading(true)
    setError(null)

    try {
      // Build the URL, adding league_id if it exists
      let url = `http://localhost:8000/api/streamers/weekly?week_offset=${weekOffset}`
      if (leagueId && swid && s2) {
        url += `&league_id=${leagueId}`
      }
      // NEW: pass the position and top-156 filters through as query params.
      // Leaving position empty means "don't filter by position at all" --
      // the backend treats a missing/blank position param as "All".
      if (position) {
        url += `&position=${position}`
      }
      if (hideTop156) {
        url += `&hide_top_156=true`
      }

      // Attach the cookies as custom headers
      const response = await fetch(url, {
        headers: {
          'x-espn-swid': swid,
          'x-espn-s2': s2
        }
      })

      if (!response.ok) {
        throw new Error("Failed to fetch data or invalid ESPN credentials")
      }

      const data = await response.json()
      setStreamers(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Re-fetch whenever the week, position, or top-156 toggle changes.
  // Note this does NOT include leagueId/swid/s2 -- those only refetch
  // when the user explicitly clicks "Save & Filter" (handleSaveAndFilter
  // below), so the app doesn't refetch on every keystroke while typing
  // credentials into the form.
  useEffect(() => {
    fetchStreamers()
  }, [weekOffset, position, hideTop156])

  // Handle saving the form
  const handleSaveAndFilter = (e) => {
    e.preventDefault()
    localStorage.setItem('espn_league_id', leagueId)
    localStorage.setItem('espn_swid', swid)
    localStorage.setItem('espn_s2', s2)
    fetchStreamers()
    setShowFilterForm(false) // collapse the form after saving
  }

  const handleClear = () => {
    localStorage.removeItem('espn_league_id')
    localStorage.removeItem('espn_swid')
    localStorage.removeItem('espn_s2')
    setLeagueId('')
    setSwid('')
    setS2('')
    setTimeout(fetchStreamers, 100)
  }

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif', maxWidth: '800px', margin: '0 auto' }}>
      <h1>Fantasy Basketball Streamer Prediction Tool</h1>

      {/* CHANGE 1: The filter form is now collapsed by default, behind a
          toggle button, instead of always taking up space at the top of
          the page. */}
      <div style={{ marginBottom: '20px' }}>
        <button
          onClick={() => setShowFilterForm(!showFilterForm)}
          style={{
            padding: '10px 16px',
            background: '#f5f5f5',
            border: '1px solid #ccc',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold',
            width: '100%',
            textAlign: 'left'
          }}
        >
          Connect Your ESPN League (to filter out rostered players) {showFilterForm ? '▲' : '▼'}
        </button>

        {showFilterForm && (
          <div style={{ background: '#f5f5f5', padding: '15px', borderRadius: '8px', marginTop: '8px' }}>
            {/* The requested description of how to find these values */}
            <p style={{ fontSize: '13px', color: '#555', marginTop: 0, marginBottom: '14px', lineHeight: '1.5' }}>
              <strong>Instructions:</strong> Step 1: Log into your ESPN Fantasy League in your browser and naviagte to the home page.<br/>
              Step 2: Open your browser's Developer Tools (usually F12), go to the
              Application (Chrome) or Storage (Firefox) tab, and look under
              Cookies for <code>espn.com</code>.<br/> 
              Step 3: Copy the values of the cookies named{' '}
              <code>SWID</code> and <code>espn_s2</code> into the fields below. Your League ID
              is the number that appears in your league's URL, e.g. <code>...leagueId=123456...</code>.
            </p>
            <form onSubmit={handleSaveAndFilter} style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <input
                type="text"
                placeholder="League ID"
                value={leagueId}
                onChange={(e) => setLeagueId(e.target.value)}
              />
              <input
                type="text"
                placeholder="SWID (include brackets, e.g. {ABCD...})"
                value={swid}
                onChange={(e) => setSwid(e.target.value)}
              />
              <input
                type="password"
                placeholder="ESPN_S2 (long string)"
                value={s2}
                onChange={(e) => setS2(e.target.value)}
              />
              <div style={{ display: 'flex', gap: '10px' }}>
                <button type="submit" style={{ padding: '8px 16px', background: '#007bff', color: 'white', border: 'none', borderRadius: '4px' }}>
                  Save & Filter
                </button>
                <button type="button" onClick={handleClear} style={{ padding: '8px 16px', background: '#ccc', border: 'none', borderRadius: '4px' }}>
                  Clear Filters
                </button>
              </div>
            </form>
          </div>
        )}
      </div>

      {/* Week Toggle (unchanged) */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
        <button
          onClick={() => setWeekOffset(0)}
          style={{ padding: '10px', background: weekOffset === 0 ? '#007bff' : '#ccc', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
        >
          Current Week
        </button>
        <button
          onClick={() => setWeekOffset(1)}
          style={{ padding: '10px', background: weekOffset === 1 ? '#007bff' : '#ccc', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
        >
          Next Week
        </button>
      </div>

      {/* CHANGE 3a: Position filter -- a row of toggle buttons, same
          pattern as the week toggle above. */}
      <div style={{ marginBottom: '10px' }}>
        <span style={{ marginRight: '8px', fontWeight: 'bold' }}>Position:</span>
        {POSITIONS.map((pos) => (
          <button
            key={pos || 'ALL'}
            onClick={() => setPosition(pos)}
            style={{
              padding: '6px 12px',
              marginRight: '6px',
              background: position === pos ? '#007bff' : '#eee',
              color: position === pos ? 'white' : '#333',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            {pos || 'All'}
          </button>
        ))}
      </div>

      {/* CHANGE 3b: Hide-top-156 toggle */}
      <div style={{ marginBottom: '20px' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontSize: '14px' }}>
          <input
            type="checkbox"
            checked={hideTop156}
            onChange={(e) => setHideTop156(e.target.checked)}
          />
          Hide last season's top-156 players (non-streamers in an average 12-person fantasy league)
        </label>
      </div>

      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      {loading ? (
        <p>Loading streamers...</p>
      ) : (
        <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse', fontSize: '13px' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #ccc', color: '#606060', fontSize: '11px', textTransform: 'uppercase' }}>
              <th style={{ padding: '8px 0' }}>Rank</th>
              <th style={{ padding: '8px 0', width: '250px' }}>Player</th>
              <th style={{ padding: '8px 0' }}>Projected Fantasy Points</th>
              <th style={{ padding: '8px 0' }}>Games</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {streamers.map((player, index) => (
              <tr
                key={index}
                onClick={() => navigate(`/player/${encodeURIComponent(player.PLAYER_NAME)}?week_offset=${weekOffset}`)}
                style={{ borderBottom: '1px solid #eee', cursor: 'pointer' }}
              >
                <td style={{ padding: '8px 0' }}>{player.RANK}</td>
                
                {/* ESPN-Style Player Cell */}
                <td style={{ padding: '8px 0', display: 'flex', alignItems: 'center', gap: '12px' }}>
    
                  {/* Circular Headshot */}
                  <div style={{
                    width: '46px', 
                    height: '46px', 
                    borderRadius: '50%', 
                    overflow: 'hidden', 
                    background: '#f0f0f0',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'flex-end',
                    flexShrink: 0
                  }}>
                    <img 
                      src={`https://cdn.nba.com/headshots/nba/latest/260x190/${player.PLAYER_ID}.png`} 
                      alt={player.PLAYER_NAME}
                      style={{ width: '130%', objectFit: 'cover' }}
                      onError={(e) => { e.target.style.display = 'none' }}
                    />
                  </div>
                  
                  {/* Stacked Text (Name on top, Team/Position below) */}
                  <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    
                    {/* Truncated Name */}
                    <span style={{ 
                      color: '#066ce4', 
                      fontSize: '15px', 
                      whiteSpace: 'nowrap', 
                      overflow: 'hidden', 
                      textOverflow: 'ellipsis',
                      maxWidth: '160px' // Adjust this width to trigger the ... earlier or later
                    }}>
                      {player.PLAYER_NAME}
                    </span>
                    
                    {/* Team and Position */}
                    <div style={{ color: '#606060', fontSize: '13px', marginTop: '1px' }}>
                      {player.TEAM_ABBREV && (
                        <span style={{ fontWeight: 'normal', marginRight: '5px' }}>
                          {player.TEAM_ABBREV}
                        </span>
                      )}
                      <span style={{ fontWeight: 'bold' }}>
                        {player.POSITION || '—'}
                      </span>
                    </div>
                    
                  </div>
                </td>

                <td style={{ padding: '8px 0' }}>{player.PREDICTED_WEEKLY_FP.toFixed(1)}</td>
                <td style={{ padding: '8px 0' }}>{player.GAMES_LOGGED} / {player.GAMES_FORECASTED}</td>
                <td style={{ textAlign: 'right', color: '#066ce4', paddingRight: '8px', fontSize: '16px' }}>›</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export default App