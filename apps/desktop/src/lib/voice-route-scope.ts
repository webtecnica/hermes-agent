import { createContext, useContext } from 'react'

/**
 * The (connection, profile) whose TTS/STT config a chat surface's voice
 * playback must resolve against.
 *
 * A Bot IS a profile: its chat's requests (model options, config reads,
 * transcripts) route through the session's OWNER ROUTE — see
 * `chat/session-tile.tsx` — not through the window's active profile. Voice
 * playback used to be the exception: it dialed `getApiRequestProfile()`, the
 * ACTIVE gateway profile, so every Bot spoke with the active profile's TTS
 * voice regardless of its own per-profile voice settings (#100864).
 *
 * Chat surfaces that belong to an owner route (Bot chats, routed sessions,
 * foreign registry sessions) provide that route here so speech is synthesized
 * with the owning profile's voice. Surfaces with no owner route leave the
 * context null and playback keeps using the active (connection, profile) —
 * the exact pre-#100864 behavior.
 */
export interface VoiceRouteScope {
  connectionId?: null | string
  profile?: null | string
}

export const VoiceRouteScopeContext = createContext<null | VoiceRouteScope>(null)

/** The current chat surface's voice route scope, or null → active scope. */
export function useVoiceRouteScope(): null | VoiceRouteScope {
  return useContext(VoiceRouteScopeContext)
}
