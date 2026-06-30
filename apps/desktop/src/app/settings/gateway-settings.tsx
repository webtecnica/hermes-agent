import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { DesktopAuthProvider, DesktopCloudAgent, DesktopConnectionProbeResult } from '@/global'
import { useI18n } from '@/i18n'
import { AlertCircle, Check, Cloud, FileText, Globe, Loader2, LogIn, Monitor, RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import { $profiles, refreshActiveProfile } from '@/store/profile'

import { CONTROL_TEXT } from './constants'
import { EmptyState, ListRow, LoadingState, Pill, SettingsContent } from './primitives'

type Mode = 'local' | 'remote' | 'cloud'
type AuthMode = 'oauth' | 'token'
type ProbeStatus = 'idle' | 'probing' | 'done' | 'error'
// Hermes Cloud discovery lifecycle for the cloud-mode panel.
type CloudDiscoverStatus = 'idle' | 'loading' | 'done' | 'error'

interface GatewaySettingsState {
  envOverride: boolean
  mode: Mode
  remoteAuthMode: AuthMode
  remoteOauthConnected: boolean
  remoteTokenPreview: string | null
  remoteTokenSet: boolean
  remoteUrl: string
}

const EMPTY_STATE: GatewaySettingsState = {
  envOverride: false,
  mode: 'local',
  remoteAuthMode: 'token',
  remoteOauthConnected: false,
  remoteTokenPreview: null,
  remoteTokenSet: false,
  remoteUrl: ''
}

function ModeCard({
  active,
  description,
  disabled,
  icon: Icon,
  onSelect,
  title
}: {
  active: boolean
  description: string
  disabled?: boolean
  icon: typeof Monitor
  onSelect: () => void
  title: string
}) {
  return (
    <button
      className={cn(
        'rounded-xl border p-3 text-left transition',
        active
          ? 'border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)'
          : 'border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) hover:bg-(--chrome-action-hover)',
        disabled && 'cursor-not-allowed opacity-50'
      )}
      disabled={disabled}
      onClick={onSelect}
      type="button"
    >
      <div className="flex items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium">
        <Icon className="size-4 text-muted-foreground" />
        <span>{title}</span>
        {active ? <Check className="ml-auto size-4 text-primary" /> : null}
      </div>
      <p className="mt-1.5 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {description}
      </p>
    </button>
  )
}

function ScopeChip({ active, label, onSelect }: { active: boolean; label: string; onSelect: () => void }) {
  return (
    <button
      className={cn(
        'rounded-full border px-3 py-1 text-[length:var(--conversation-caption-font-size)] transition',
        active
          ? 'border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary) text-(--ui-text-primary)'
          : 'border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover)'
      )}
      onClick={onSelect}
      type="button"
    >
      {label}
    </button>
  )
}

export function GatewaySettings() {
  const { t } = useI18n()
  const g = t.settings.gateway
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [signingIn, setSigningIn] = useState(false)
  const [state, setState] = useState<GatewaySettingsState>(EMPTY_STATE)
  const [remoteToken, setRemoteToken] = useState('')
  const [lastTest, setLastTest] = useState<null | string>(null)

  // --- Hermes Cloud (cloud mode) state ---
  // One portal session powers discovery + the silent per-agent cascade. These
  // track the cloud panel: whether we're signed in, the discovered agent list,
  // and which agent is mid-connect.
  const [cloudSignedIn, setCloudSignedIn] = useState(false)
  const [cloudSigningIn, setCloudSigningIn] = useState(false)
  const [cloudAgents, setCloudAgents] = useState<DesktopCloudAgent[]>([])
  const [cloudDiscover, setCloudDiscover] = useState<CloudDiscoverStatus>('idle')
  const [cloudConnectingId, setCloudConnectingId] = useState<null | string>(null)

  // Connection scope: null = the global/default connection (the original
  // behavior); a profile name = that profile's per-profile remote override, so
  // each profile can point at its own backend.
  const [scope, setScope] = useState<null | string>(null)
  const profiles = useStore($profiles)

  useEffect(() => {
    void refreshActiveProfile()
  }, [])

  // Auth-mode probe: as the user types a remote URL we ask the gateway (via
  // its public /api/status) whether it gates with OAuth or a static session
  // token, so we can show the right control (login button vs token box).
  const [probeStatus, setProbeStatus] = useState<ProbeStatus>('idle')
  const [probe, setProbe] = useState<DesktopConnectionProbeResult | null>(null)
  const probeSeq = useRef(0)

  useEffect(() => {
    let cancelled = false
    const desktop = window.hermesDesktop

    if (!desktop?.getConnectionConfig) {
      setLoading(false)

      return () => void (cancelled = true)
    }

    setLoading(true)
    // Clear scope-local entry state so a token from one scope can't leak into
    // the next when switching profiles.
    setRemoteToken('')
    setLastTest(null)

    desktop
      .getConnectionConfig(scope)
      .then(config => {
        if (cancelled) {
          return
        }

        setState(config)
      })
      .catch(err => notifyError(err, g.failedLoad))
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => void (cancelled = true)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reload on scope change only; copy is stable
  }, [scope])

  // Debounced probe of the entered remote URL. Only runs in remote mode with a
  // syntactically plausible URL. The probe result drives whether we render the
  // OAuth login button or the session-token entry box. The effective auth mode
  // prefers a fresh probe result over the saved value.
  const trimmedUrl = state.remoteUrl.trim()
  useEffect(() => {
    if (state.mode !== 'remote' || !trimmedUrl || !/^https?:\/\//i.test(trimmedUrl)) {
      setProbeStatus('idle')
      setProbe(null)

      return
    }

    const desktop = window.hermesDesktop

    if (!desktop?.probeConnectionConfig) {
      return
    }

    const seq = ++probeSeq.current
    setProbeStatus('probing')

    const timer = setTimeout(() => {
      desktop
        .probeConnectionConfig(trimmedUrl)
        .then(result => {
          if (seq !== probeSeq.current) {
            return
          }

          setProbe(result)
          setProbeStatus(result.reachable ? 'done' : 'error')
        })
        .catch(() => {
          if (seq !== probeSeq.current) {
            return
          }

          setProbe(null)
          setProbeStatus('error')
        })
    }, 500)

    return () => clearTimeout(timer)
  }, [state.mode, trimmedUrl])

  // Effective auth mode: a reachable probe wins; otherwise fall back to the
  // saved config's mode so a re-open of settings doesn't flicker.
  const authMode: AuthMode = useMemo(() => {
    if (probeStatus === 'done' && probe && probe.authMode !== 'unknown') {
      return probe.authMode
    }

    return state.remoteAuthMode
  }, [probe, probeStatus, state.remoteAuthMode])

  // Whether we actually KNOW how this gateway authenticates yet. Until we do,
  // neither the OAuth button nor the session-token box should render —
  // `authMode` defaults to 'token', so without this gate the token box flashes
  // for every gateway (including OAuth ones) during the idle/probing window
  // before the first probe lands. The scheme is known when either:
  //   * the live probe finished (probeStatus 'done'), or
  //   * we're idle but showing a previously-saved remote config (re-opening
  //     settings for a gateway already signed-in or with a saved token), so
  //     its control appears immediately with no flicker.
  // While probing (or after a probe error), the scheme is unknown and we show
  // the probe status row instead of a control.
  const hasSavedRemote = state.remoteTokenSet || state.remoteOauthConnected

  const authResolved = useMemo(() => {
    if (probeStatus === 'done') {
      return true
    }

    return probeStatus === 'idle' && hasSavedRemote
  }, [probeStatus, hasSavedRemote])

  const providerLabel = useMemo(() => {
    const providers: DesktopAuthProvider[] = probe?.providers ?? []

    if (providers.length === 1) {
      return providers[0].displayName || providers[0].name
    }

    if (providers.length > 1) {
      return providers.map(p => p.displayName || p.name).join(' / ')
    }

    return t.boot.failure.identityProvider
  }, [probe, t.boot.failure.identityProvider])

  // A username/password gateway authenticates through a credential form on the
  // gateway's /login page (POST /auth/password-login) rather than an OAuth
  // redirect. Everything downstream — the session cookie, the ws-ticket mint,
  // the persistent partition — is identical, so the desktop drives it through
  // the same sign-in window; only the button copy changes. We treat the
  // gateway as password-style only when EVERY advertised provider supports
  // password, so a mixed deployment keeps the generic OAuth copy.
  const isPasswordProvider = useMemo(() => {
    const providers: DesktopAuthProvider[] = probe?.providers ?? []

    return providers.length > 0 && providers.every(p => p.supportsPassword)
  }, [probe])

  // The 'default' profile uses the global ("All profiles") connection, so the
  // per-profile scopes are the named, non-default profiles.
  const namedProfiles = useMemo(() => profiles.filter(profile => profile.name !== 'default'), [profiles])

  const oauthConnected = state.remoteOauthConnected

  const canUseRemote = useMemo(() => {
    if (!trimmedUrl) {
      return false
    }

    if (authMode === 'oauth') {
      return oauthConnected
    }

    return Boolean(remoteToken.trim()) || state.remoteTokenSet
  }, [authMode, oauthConnected, remoteToken, state.remoteTokenSet, trimmedUrl])

  const payload = () => ({
    mode: state.mode,
    profile: scope ?? undefined,
    remoteAuthMode: authMode,
    remoteToken: authMode === 'token' ? remoteToken.trim() || undefined : undefined,
    remoteUrl: trimmedUrl
  })

  const save = async (apply: boolean) => {
    if (state.mode === 'remote' && !canUseRemote) {
      notify({
        kind: 'warning',
        title: g.incompleteTitle,
        message: authMode === 'oauth' ? g.incompleteSignIn : g.incompleteToken
      })

      return
    }

    setSaving(true)

    try {
      const next = apply
        ? await window.hermesDesktop.applyConnectionConfig(payload())
        : await window.hermesDesktop.saveConnectionConfig(payload())

      setState(next)
      setRemoteToken('')
      notify({
        kind: 'success',
        title: apply ? g.restartingTitle : g.savedTitle,
        message: apply ? g.restartingMessage : g.savedMessage
      })
    } catch (err) {
      notifyError(err, apply ? g.applyFailed : g.saveFailed)
    } finally {
      setSaving(false)
    }
  }

  // OAuth sign-in: persist the URL + oauth mode first (so the saved config has
  // the URL the login window needs), then open the gateway login window and
  // refresh the connection status from the saved config once it completes.
  const signIn = async () => {
    if (!trimmedUrl) {
      notify({ kind: 'warning', title: g.incompleteTitle, message: g.enterUrlFirst })

      return
    }

    setSigningIn(true)

    try {
      // Save (don't apply/restart) so the login window has a URL to use and the
      // oauth mode is persisted, without yet flipping the live connection.
      const saved = await window.hermesDesktop.saveConnectionConfig({
        mode: state.mode,
        profile: scope ?? undefined,
        remoteAuthMode: 'oauth',
        remoteUrl: trimmedUrl
      })

      setState(saved)

      const result = await window.hermesDesktop.oauthLoginConnectionConfig(trimmedUrl)

      if (result.connected) {
        const refreshed = await window.hermesDesktop.getConnectionConfig(scope)
        setState(refreshed)
        notify({ kind: 'success', title: g.signedIn, message: g.connectedTo(providerLabel) })
      } else {
        notify({
          kind: 'warning',
          title: t.boot.failure.signInIncompleteTitle,
          message: t.boot.failure.signInIncompleteMessage
        })
      }
    } catch (err) {
      notifyError(err, g.signInFailed)
    } finally {
      setSigningIn(false)
    }
  }

  const signOut = async () => {
    setSigningIn(true)

    try {
      await window.hermesDesktop.oauthLogoutConnectionConfig(trimmedUrl || undefined)
      const refreshed = await window.hermesDesktop.getConnectionConfig(scope)
      setState(refreshed)
      notify({ kind: 'success', title: g.signedOutTitle, message: g.signedOutMessage })
    } catch (err) {
      notifyError(err, g.signOutFailed)
    } finally {
      setSigningIn(false)
    }
  }

  // --- Hermes Cloud handlers ---

  // Pull the discovered agent list over the shared portal session. Tolerant of
  // a lapsed session: a needsCloudLogin error flips us back to signed-out.
  const discoverCloud = async () => {
    const desktop = window.hermesDesktop

    if (!desktop?.cloud) {
      return
    }

    setCloudDiscover('loading')

    try {
      const { agents } = await desktop.cloud.discover()
      setCloudAgents(agents)
      setCloudDiscover('done')
    } catch (err) {
      setCloudAgents([])
      setCloudDiscover('error')

      // A lapsed/absent portal session means we're effectively signed out.
      if (err && typeof err === 'object' && 'needsCloudLogin' in err) {
        setCloudSignedIn(false)
      }

      notifyError(err, g.cloudDiscoverFailed)
    }
  }

  // On entering cloud mode (or scope change), read the portal session status and
  // auto-discover when already signed in, so the picker is populated on open.
  useEffect(() => {
    if (state.mode !== 'cloud') {
      return
    }

    const desktop = window.hermesDesktop

    if (!desktop?.cloud) {
      return
    }

    let cancelled = false
    desktop.cloud
      .status()
      .then(status => {
        if (cancelled) {
          return
        }

        setCloudSignedIn(status.signedIn)

        if (status.signedIn) {
          void discoverCloud()
        } else {
          setCloudAgents([])
          setCloudDiscover('idle')
        }
      })
      .catch(() => {
        if (!cancelled) {
          setCloudSignedIn(false)
        }
      })

    return () => void (cancelled = true)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reload on mode/scope change only
  }, [state.mode, scope])

  const cloudSignIn = async () => {
    const desktop = window.hermesDesktop

    if (!desktop?.cloud) {
      return
    }

    setCloudSigningIn(true)

    try {
      const result = await desktop.cloud.login()
      setCloudSignedIn(result.signedIn)

      if (result.signedIn) {
        await discoverCloud()
      }
    } catch (err) {
      notifyError(err, g.cloudSignInFailed)
    } finally {
      setCloudSigningIn(false)
    }
  }

  const cloudSignOut = async () => {
    const desktop = window.hermesDesktop

    if (!desktop?.cloud) {
      return
    }

    setCloudSigningIn(true)

    try {
      await desktop.cloud.logout()
      setCloudSignedIn(false)
      setCloudAgents([])
      setCloudDiscover('idle')
      notify({ kind: 'success', title: g.cloudSignedOutTitle, message: g.cloudSignedOutMessage })
    } catch (err) {
      notifyError(err, g.signOutFailed)
    } finally {
      setCloudSigningIn(false)
    }
  }

  // Select a discovered agent: drive the silent per-agent cascade (no second
  // prompt — the shared portal session auto-approves), then persist a cloud-mode
  // connection pointed at its dashboardUrl and apply it (reconnects the window).
  const connectCloudAgent = async (agent: DesktopCloudAgent) => {
    if (!agent.dashboardUrl) {
      return
    }

    const desktop = window.hermesDesktop

    if (!desktop?.cloud) {
      return
    }

    setCloudConnectingId(agent.id)

    try {
      const result = await desktop.cloud.agentSignIn(agent.dashboardUrl)

      if (!result.connected) {
        notify({
          kind: 'warning',
          title: t.boot.failure.signInIncompleteTitle,
          message: t.boot.failure.signInIncompleteMessage
        })

        return
      }

      // Persist a cloud-mode connection (remote-shaped, oauth) and reconnect.
      const next = await desktop.applyConnectionConfig({
        mode: 'cloud',
        profile: scope ?? undefined,
        remoteAuthMode: 'oauth',
        remoteUrl: agent.dashboardUrl
      })

      setState(next)
      notify({ kind: 'success', title: g.cloudConnectedTitle, message: g.cloudConnectedTo(agent.name) })
    } catch (err) {
      if (err && typeof err === 'object' && 'needsCloudLogin' in err) {
        setCloudSignedIn(false)
      }

      notifyError(err, g.cloudConnectFailed)
    } finally {
      setCloudConnectingId(null)
    }
  }

  const testRemote = async () => {
    if (!canUseRemote) {
      notify({
        kind: 'warning',
        title: g.incompleteTitle,
        message: authMode === 'oauth' ? g.incompleteSignInTest : g.incompleteTokenTest
      })

      return
    }

    setTesting(true)
    setLastTest(null)

    try {
      const result = await window.hermesDesktop.testConnectionConfig({
        mode: 'remote',
        profile: scope ?? undefined,
        remoteAuthMode: authMode,
        remoteToken: authMode === 'token' ? remoteToken.trim() || undefined : undefined,
        remoteUrl: trimmedUrl
      })

      const message = g.connectedTo(result.baseUrl, result.version ?? undefined)
      setLastTest(message)
      notify({ kind: 'success', title: g.reachableTitle, message })
    } catch (err) {
      notifyError(err, g.testFailed)
    } finally {
      setTesting(false)
    }
  }

  if (loading) {
    return <LoadingState label={g.loading} />
  }

  if (!window.hermesDesktop?.getConnectionConfig) {
    return <EmptyState description={g.unavailableDesc} title={g.unavailableTitle} />
  }

  return (
    <SettingsContent>
      <div className="mb-5">
        <div className="flex items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium">
          <Globe className="size-4 text-muted-foreground" />
          {g.title}
          {state.envOverride ? <Pill tone="primary">{g.envOverride}</Pill> : null}
        </div>
        <p className="mt-2 max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {g.intro}
        </p>
      </div>

      {namedProfiles.length > 0 ? (
        <div className="mb-5 grid gap-2">
          <div className="text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
            {g.appliesTo}
          </div>
          <div className="flex flex-wrap gap-1.5">
            <ScopeChip active={scope === null} label={g.allProfiles} onSelect={() => setScope(null)} />
            {namedProfiles.map(profile => (
              <ScopeChip
                active={scope === profile.name}
                key={profile.name}
                label={profile.name}
                onSelect={() => setScope(profile.name)}
              />
            ))}
          </div>
          <p className="text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {scope === null ? g.defaultConnection : g.profileConnection(scope)}
          </p>
        </div>
      ) : null}

      {state.envOverride ? (
        <div className="mb-5 flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2.5 text-[length:var(--conversation-caption-font-size)] text-destructive">
          <AlertCircle className="mt-0.5 size-4 shrink-0" />
          <div>
            <div className="font-medium">{g.envOverrideTitle}</div>
            <div className="mt-1 leading-5">{g.envOverrideDesc}</div>
          </div>
        </div>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-3">
        <ModeCard
          active={state.mode === 'local'}
          description={g.localDesc}
          disabled={state.envOverride}
          icon={Monitor}
          onSelect={() => setState(current => ({ ...current, mode: 'local' }))}
          title={g.localTitle}
        />
        <ModeCard
          active={state.mode === 'cloud'}
          description={g.cloudDesc}
          disabled={state.envOverride}
          icon={Cloud}
          onSelect={() => setState(current => ({ ...current, mode: 'cloud' }))}
          title={g.cloudTitle}
        />
        <ModeCard
          active={state.mode === 'remote'}
          description={g.remoteDesc}
          disabled={state.envOverride}
          icon={Globe}
          onSelect={() => setState(current => ({ ...current, mode: 'remote' }))}
          title={g.remoteTitle}
        />
      </div>

      {/* Hermes Cloud panel: one portal sign-in, then a discovered-agent picker
          whose selection drives the silent per-agent cascade + a cloud
          connection. Replaces the URL/token form while in cloud mode. */}
      {state.mode === 'cloud' && !state.envOverride ? (
        <div className="mt-5 grid gap-1">
          <ListRow
            action={
              cloudSignedIn ? (
                <div className="flex items-center gap-2">
                  <Pill tone="primary">
                    <Check className="size-3" /> {g.cloudSignedIn}
                  </Pill>
                  <Button disabled={cloudSigningIn} onClick={() => void cloudSignOut()} variant="outline">
                    {cloudSigningIn ? <Loader2 className="animate-spin" /> : null}
                    {g.signOut}
                  </Button>
                </div>
              ) : (
                <Button disabled={cloudSigningIn} onClick={() => void cloudSignIn()}>
                  {cloudSigningIn ? <Loader2 className="animate-spin" /> : <LogIn />}
                  {g.cloudSignIn}
                </Button>
              )
            }
            description={cloudSignedIn ? g.cloudSignedInDesc : g.cloudNeedsSignIn}
            title={g.cloudSignInTitle}
          />

          {cloudSignedIn ? (
            <div className="mt-3">
              <div className="mb-2 flex items-center justify-between">
                <div className="text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
                  {g.cloudAgentsTitle}
                </div>
                <Button
                  disabled={cloudDiscover === 'loading'}
                  onClick={() => void discoverCloud()}
                  size="sm"
                  variant="text"
                >
                  {cloudDiscover === 'loading' ? <Loader2 className="animate-spin" /> : <RefreshCw />}
                  {g.cloudRefresh}
                </Button>
              </div>

              {cloudDiscover === 'loading' ? (
                <div className="flex items-center gap-2 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                  <Loader2 className="size-4 animate-spin" />
                  {g.cloudLoadingAgents}
                </div>
              ) : cloudAgents.length === 0 ? (
                <div className="flex items-start gap-2 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                  <AlertCircle className="mt-0.5 size-4 shrink-0" />
                  {g.cloudNoAgents}
                </div>
              ) : (
                <div className="grid gap-1">
                  {cloudAgents.map(agent => (
                    <ListRow
                      action={
                        <Button
                          disabled={!agent.dashboardUrl || cloudConnectingId !== null}
                          onClick={() => void connectCloudAgent(agent)}
                          size="sm"
                        >
                          {cloudConnectingId === agent.id ? <Loader2 className="animate-spin" /> : null}
                          {agent.dashboardUrl
                            ? cloudConnectingId === agent.id
                              ? g.cloudConnecting
                              : g.cloudConnect
                            : g.cloudAgentProvisioning}
                        </Button>
                      }
                      description={g.cloudStatusLabel(agent.dashboardGatewayState)}
                      key={agent.id}
                      title={agent.name}
                    />
                  ))}
                </div>
              )}
            </div>
          ) : null}
        </div>
      ) : null}

      {state.mode === 'remote' && !state.envOverride ? (
      <div className="mt-5 grid gap-1">
        <ListRow
          action={
            <Input
              className={cn('h-8', CONTROL_TEXT)}
              disabled={state.envOverride}
              onChange={event => setState(current => ({ ...current, remoteUrl: event.target.value }))}
              placeholder="https://gateway.example.com/hermes"
              value={state.remoteUrl}
            />
          }
          description={g.remoteUrlDesc}
          title={g.remoteUrlTitle}
        />

        {state.mode === 'remote' && probeStatus === 'probing' ? (
          <div className="flex items-center gap-2 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            <Loader2 className="size-4 animate-spin" />
            {g.probing}
          </div>
        ) : null}

        {state.mode === 'remote' && probeStatus === 'error' ? (
          <div className="flex items-start gap-2 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            {g.probeError}
          </div>
        ) : null}

        {/* OAuth / password gateways: present a sign-in button + connection status. */}
        {state.mode === 'remote' && authResolved && authMode === 'oauth' ? (
          <ListRow
            action={
              oauthConnected ? (
                <div className="flex items-center gap-2">
                  <Pill tone="primary">
                    <Check className="size-3" /> {g.signedIn}
                  </Pill>
                  <Button disabled={signingIn || state.envOverride} onClick={() => void signOut()} variant="outline">
                    {signingIn ? <Loader2 className="animate-spin" /> : null}
                    {g.signOut}
                  </Button>
                </div>
              ) : (
                <Button disabled={signingIn || state.envOverride || !trimmedUrl} onClick={() => void signIn()}>
                  {signingIn ? <Loader2 className="animate-spin" /> : <LogIn />}
                  {isPasswordProvider ? g.signIn : g.signInWith(providerLabel)}
                </Button>
              )
            }
            description={
              oauthConnected
                ? isPasswordProvider
                  ? g.authSignedInPassword
                  : g.authSignedInOauth
                : isPasswordProvider
                  ? g.authNeedsPassword
                  : g.authNeedsOauth(providerLabel)
            }
            title={g.authTitle}
          />
        ) : null}

        {/* Session-token gateways: keep the existing token entry box. */}
        {state.mode === 'remote' && authResolved && authMode === 'token' ? (
          <ListRow
            action={
              <Input
                autoComplete="off"
                className={cn('h-8 font-mono', CONTROL_TEXT)}
                disabled={state.envOverride}
                onChange={event => setRemoteToken(event.target.value)}
                placeholder={
                  state.remoteTokenSet ? g.existingToken(state.remoteTokenPreview ?? g.savedToken) : g.pasteSessionToken
                }
                type="password"
                value={remoteToken}
              />
            }
            description={g.tokenDesc}
            title={g.tokenTitle}
          />
        ) : null}
      </div>
      ) : null}

      {lastTest ? <div className="mt-4 text-xs text-primary">{lastTest}</div> : null}

      {/* Test/Save apply to local + remote. Cloud connects via the agent picker
          above (which applies a cloud connection on select), so its only
          bottom-row action would be redundant — hidden in cloud mode. */}
      {state.mode !== 'cloud' ? (
        <div className="mt-6 flex flex-wrap items-center justify-end gap-4">
          {state.mode === 'remote' ? (
            <Button
              className="mr-auto"
              disabled={state.envOverride || testing || !canUseRemote}
              onClick={() => void testRemote()}
              size="sm"
              variant="text"
            >
              {testing ? <Loader2 className="animate-spin" /> : null}
              {g.testRemote}
            </Button>
          ) : null}
          <Button disabled={state.envOverride || saving} onClick={() => void save(false)} size="sm" variant="textStrong">
            {g.saveForRestart}
          </Button>
          <Button disabled={state.envOverride || saving} onClick={() => void save(true)} size="sm">
            {saving ? <Loader2 className="animate-spin" /> : null}
            {g.saveAndReconnect}
          </Button>
        </div>
      ) : null}

      <div className="mt-6 grid gap-1">
        <ListRow
          action={
            <Button onClick={() => void window.hermesDesktop?.revealLogs()} size="sm" variant="textStrong">
              <FileText />
              {g.openLogs}
            </Button>
          }
          description={g.diagnosticsDesc}
          title={g.diagnostics}
        />
      </div>
    </SettingsContent>
  )
}
