import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import App from './App.jsx'
import PlayerProfile from './PlayerProfile.jsx' // We will create this next

ReactDOM.createRoot(document.getElementById('root')).render(
  <BrowserRouter>
    <Routes>
      {/* Your existing leaderboard becomes the home page */}
      <Route path="/" element={<App />} />
      
      {/* The dynamic route for individual players */}
      <Route path="/player/:playerName" element={<PlayerProfile />} />
    </Routes>
  </BrowserRouter>
)