import { useState, useEffect } from 'react'
import { useParams, useNavigate, useLocation } from 'react-router-dom'

function PlayerProfile() {
  const { playerName } = useParams()
  const navigate = useNavigate()
  const location = useLocation()

  // Extract the week_offset from the URL, default to 0
  const queryParams = new URLSearchParams(location.search)
  const weekOffset = queryParams.get('week_offset') || 0

  const [playerData, setPlayerData] = useState(null)
  const [error, setError] = useState(null)
  const [expandedGameDate, setExpandedGameDate] = useState(null)

  useEffect(() => {
    fetch(`http://localhost:8000/api/player/${encodeURIComponent(playerName)}?week_offset=${weekOffset}`)
      .then(res => {
        if (!res.ok) throw new Error("Failed to load player details")
        return res.json()
      })
      .then(data => setPlayerData(data))
      .catch(err => setError(err.message))
  }, [playerName, weekOffset])

  const toggleGame = (date) => {
    setExpandedGameDate(prev => (prev === date ? null : date))
  }

  if (error) {
    return (
      <div style={{ padding: '20px', fontFamily: "'Helvetica Neue', Helvetica, Arial, sans-serif" }}>
        <button onClick={() => navigate(-1)} style={{ marginBottom: '15px', cursor: 'pointer' }}>⬅ Back</button>
        <p style={{ color: 'red' }}>Error: {error}</p>
      </div>
    )
  }

  if (!playerData) {
    return (
      <h2 style={{ padding: '20px', fontFamily: "'Helvetica Neue', Helvetica, Arial, sans-serif" }}>
        Loading {playerName}...
      </h2>
    )
  }

  return (
    <div style={{ padding: '20px', fontFamily: "'Helvetica Neue', Helvetica, Arial, sans-serif", maxWidth: '650px', margin: '0 auto' }}>
      <button 
        onClick={() => navigate(-1)} 
        style={{ 
          padding: '8px 14px', 
          marginBottom: '20px', 
          cursor: 'pointer',
          borderRadius: '4px',
          border: '1px solid #ccc',
          background: '#fff'
        }}
      >
        ⬅ Back to Leaderboard
      </button>

      {/* Header with Circular Headshot and Player Name */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '24px' }}>
        <div style={{
          width: '70px',
          height: '70px',
          borderRadius: '50%',
          overflow: 'hidden',
          background: '#f0f0f0',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'flex-end',
          flexShrink: 0,
          border: '1px solid #e0e0e0'
        }}>
          <img 
            src={`https://cdn.nba.com/headshots/nba/latest/260x190/${playerData.player_id}.png`} 
            alt={playerData.player_name}
            style={{ width: '130%', objectFit: 'cover' }}
            onError={(e) => { e.target.style.display = 'none' }}
          />
        </div>
        <h1 style={{ margin: 0, fontSize: '28px', fontWeight: 'bold' }}>
          {playerData.player_name}
        </h1>
      </div>

      {/* Recent Performance Card */}
      <div style={{ background: '#f8f9fa', padding: '16px', borderRadius: '8px', marginBottom: '24px', border: '1px solid #eee' }}>
        <h3 style={{ margin: '0 0 8px 0', fontSize: '16px', color: '#333' }}>Recent Performance</h3>
        <p style={{ margin: 0, fontSize: '15px' }}>
          <strong>Past 10 Games Avg:</strong>{' '}
          {typeof playerData.actual_10_game_avg === 'number'
            ? playerData.actual_10_game_avg.toFixed(1)
            : 'N/A'}{' '}
          Fantasy Points
        </p>
      </div>

      <h3 style={{ fontSize: '18px', marginBottom: '12px' }}>This Week's Schedule & Projections</h3>
      
      {/* Schedule & Projections List */}
      {playerData.upcoming_games && playerData.upcoming_games.length > 0 ? (
        <div style={{ borderTop: '1px solid #e0e0e0' }}>
          {playerData.upcoming_games.map((game, index) => {
            const proj = game.ML_PREDICTION ?? game.PREDICTED_FP
            const isExpanded = expandedGameDate === game.GAME_DATE

            return (
              <div key={index} style={{ borderBottom: '1px solid #e0e0e0' }}>
                {/* Game Row (Clickable) */}
                <div 
                  onClick={() => toggleGame(game.GAME_DATE)}
                  style={{ 
                    padding: '14px 10px', 
                    display: 'flex', 
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    cursor: 'pointer',
                    background: isExpanded ? '#f4f6f8' : 'transparent',
                    transition: 'background 0.15s ease'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ fontWeight: 'bold', fontSize: '14px' }}>{game.GAME_DATE}</span>
                    {game.OPPONENT && (
                      <span style={{ color: '#606060', fontSize: '13px' }}>
                        vs {game.OPPONENT}
                      </span>
                    )}
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span style={{ fontSize: '14px' }}>
                      Projected: <strong>{typeof proj === 'number' ? proj.toFixed(1) : 'N/A'}</strong> Fantasy Points
                    </span>
                    <span style={{ fontSize: '11px', color: '#888', width: '12px', textAlign: 'center' }}>
                      {isExpanded ? '▲' : '▼'}
                    </span>
                  </div>
                </div>

                {/* Dropdown Details: Baseline Stats */}
                {isExpanded && playerData.baseline_stats && (
                  <div style={{ 
                    padding: '12px 14px 16px 14px', 
                    background: '#f8f9fa', 
                    borderTop: '1px dashed #e0e0e0',
                    fontSize: '13px'
                  }}>
                    <div style={{ color: '#606060', fontWeight: 'bold', marginBottom: '8px', textTransform: 'uppercase', fontSize: '11px' }}>
                      Past 4-Game Rolling Averages
                    </div>
                    <div style={{ 
                      display: 'grid', 
                      gridTemplateColumns: 'repeat(auto-fit, minmax(75px, 1fr))', 
                      gap: '8px', 
                      color: '#222' 
                    }}>
                      <div><span style={{ color: '#666' }}>MIN:</span> <strong>{playerData.baseline_stats.MIN?.toFixed(1) ?? '—'}</strong></div>
                      <div><span style={{ color: '#666' }}>PTS:</span> <strong>{playerData.baseline_stats.PTS?.toFixed(1) ?? '—'}</strong></div>
                      <div><span style={{ color: '#666' }}>REB:</span> <strong>{playerData.baseline_stats.REB?.toFixed(1) ?? '—'}</strong></div>
                      <div><span style={{ color: '#666' }}>AST:</span> <strong>{playerData.baseline_stats.AST?.toFixed(1) ?? '—'}</strong></div>
                      <div><span style={{ color: '#666' }}>STL:</span> <strong>{playerData.baseline_stats.STL?.toFixed(1) ?? '—'}</strong></div>
                      <div><span style={{ color: '#666' }}>BLK:</span> <strong>{playerData.baseline_stats.BLK?.toFixed(1) ?? '—'}</strong></div>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      ) : (
        <p style={{ color: '#666', fontSize: '14px' }}>No games scheduled for this week.</p>
      )}
    </div>
  )
}

export default PlayerProfile