import { useStore } from '@nanostores/react'
import { useCallback, useMemo } from 'react'

import type { CommandCenterSection } from '@/app/command-center'
import { $terminalTakeover, setTerminalTakeover } from '@/app/right-sidebar/store'
import { GatewayMenuPanel } from '@/app/shell/gateway-menu-panel'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { useI18n } from '@/i18n'
import {
  Activity,
  AlertCircle,
  Clock,
  Command,
  Hash,
  Loader2,
  Network,
  Sparkles,
  Terminal,
  Zap,
  ZapFilled
} from '@/lib/icons'
import type { RuntimeReadinessResult } from '@/lib/runtime-readiness'
import { contextBarLabel, LiveDuration, usageContextLabel } from '@/lib/statusbar'
import { cn } from '@/lib/utils'
import { setGlobalYolo, setSessionYolo } from '@/lib/yolo-session'
import { $desktopActionTasks } from '@/store/activity'
import { $previewServerRestartStatus } from '@/store/preview'
import {
  $activeSessionId,
  $busy,
  $connection,
  $currentUsage,
  $sessionStartedAt,
  $turnStartedAt,
  $workingSessionIds,
  $yoloActive,
  setYoloActive
} from '@/store/session'
import { $subagentsBySession, activeSubagentCount } from '@/store/subagents'
import { $gatewayRestarting } from '@/store/system-actions'
import {
  $backendUpdateApply,
  $backendUpdateStatus,
  $desktopVersion,
  $updateApply,
  $updateStatus,
  openUpdateOverlayFor
} from '@/store/updates'
import type { StatusResponse } from '@/types/hermes'

import { CRON_ROUTE, SETTINGS_ROUTE } from '../../routes'
import type { StatusbarItem, StatusbarSelectModifiers } from '../statusbar-controls'

interface StatusbarItemsOptions {
  agentsOpen: boolean
  chatOpen: boolean
  commandCenterOpen: boolean
  extraLeftItems: readonly StatusbarItem[]
  extraRightItems: readonly StatusbarItem[]
  gatewayLogLines: readonly string[]
  gatewayState: string
  inferenceStatus: RuntimeReadinessResult | null
  openAgents: () => void
  openCommandCenterSection: (section: CommandCenterSection) => void
  freshDraftReady: boolean
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
  statusSnapshot: StatusResponse | null
  toggleCommandCenter: () => void
}

export function useStatusbarItems({
  agentsOpen,
  chatOpen,
  commandCenterOpen,
  extraLeftItems,
  extraRightItems,
  gatewayLogLines,
  gatewayState,
  inferenceStatus,
  openAgents,
  openCommandCenterSection,
  freshDraftReady,
  requestGateway,
  statusSnapshot,
  toggleCommandCenter
}: StatusbarItemsOptions) {
  const { t } = useI18n()
  const copy = t.shell.statusbar
  const activeSessionId = useStore($activeSessionId)
  const terminalTakeover = useStore($terminalTakeover)
  const yoloActive = useStore($yoloActive)
  const busy = useStore($busy)
  const currentUsage = useStore($currentUsage)
  const desktopActionTasks = useStore($desktopActionTasks)
  const gatewayRestarting = useStore($gatewayRestarting)
  const previewServerRestartStatus = useStore($previewServerRestartStatus)
  const sessionStartedAt = useStore($sessionStartedAt)
  const turnStartedAt = useStore($turnStartedAt)
  const workingSessionIds = useStore($workingSessionIds)
  const subagentsBySession = useStore($subagentsBySession)
  const updateStatus = useStore($updateStatus)
  const updateApply = useStore($updateApply)
  const backendUpdateStatus = useStore($backendUpdateStatus)
  const backendUpdateApply = useStore($backendUpdateApply)
  const desktopVersion = useStore($desktopVersion)
  const connection = useStore($connection)

  const contextUsage = useMemo(() => usageContextLabel(currentUsage), [currentUsage])
  const contextBar = useMemo(() => contextBarLabel(currentUsage), [currentUsage])

  // Per-session approval bypass (same scope as the TUI's Shift+Tab). On a
  // new-chat draft (no runtime session yet) we arm locally; the session-create
  // path applies it once the backend session exists.
  //
  // Shift+click flips the GLOBAL approvals.mode instead — a persistent,
  // all-sessions/CLI/TUI/cron bypass that survives restarts.
  const toggleYolo = useCallback(
    async (modifiers?: StatusbarSelectModifiers) => {
      const next = !$yoloActive.get()

      setYoloActive(next)

      if (modifiers?.shiftKey) {
        try {
          await setGlobalYolo(requestGateway, next)
        } catch {
          setYoloActive(!next)
        }

        return
      }

      const sid = $activeSessionId.get()

      if (!sid) {
        return
      }

      try {
        await setSessionYolo(requestGateway, sid, next)
      } catch {
        setYoloActive(!next)
      }
    },
    [requestGateway]
  )

  const showYoloToggle = gatewayState === 'open' && (!!activeSessionId || freshDraftReady)

  const gatewayMenuContent = useMemo(
    () => (
      <GatewayMenuPanel
        gatewayState={gatewayState}
        inferenceStatus={inferenceStatus}
        logLines={gatewayLogLines}
        onOpenSystem={() => openCommandCenterSection('system')}
        statusSnapshot={statusSnapshot}
      />
    ),
    [gatewayLogLines, gatewayState, inferenceStatus, openCommandCenterSection, statusSnapshot]
  )

  const { bgFailed, bgRunning, subagentsRunning } = useMemo(() => {
    const actions = Object.values(desktopActionTasks)
    const running = actions.filter(t => t.status.running).length
    const failed = actions.filter(t => !t.status.running && (t.status.exit_code ?? 0) !== 0).length
    const previewRunning = previewServerRestartStatus === 'running' ? 1 : 0
    const previewFailed = previewServerRestartStatus === 'error' ? 1 : 0

    const subagentsRunning = Object.values(subagentsBySession).reduce(
      (sum, items) => sum + activeSubagentCount(items),
      0
    )

    return {
      bgFailed: failed + previewFailed,
      bgRunning: workingSessionIds.length + running + previewRunning,
      subagentsRunning
    }
  }, [desktopActionTasks, previewServerRestartStatus, subagentsBySession, workingSessionIds])

  const gatewayOpen = gatewayState === 'open'
  const gatewayConnecting = gatewayState === 'connecting'
  const inferenceReady = gatewayOpen && inferenceStatus?.ready === true
  const gatewayDegraded = gatewayOpen || gatewayConnecting

  const gatewayDetail = gatewayOpen
    ? inferenceStatus?.ready
      ? copy.gatewayReady
      : inferenceStatus
        ? copy.gatewayNeedsSetup
        : copy.gatewayChecking
    : gatewayConnecting
      ? copy.gatewayConnecting
      : copy.gatewayOffline

  const gatewayClassName = inferenceReady
    ? undefined
    : gatewayDegraded
      ? 'text-amber-600 hover:text-amber-600'
      : 'text-destructive hover:text-destructive'

  const clientVersionItem = useMemo<StatusbarItem>(() => {
    const appVersion = desktopVersion?.appVersion
    const sha = updateStatus?.currentSha?.slice(0, 7) ?? null
    const behind = updateStatus?.behind ?? 0
    const applying = updateApply.applying || updateApply.stage === 'restart'
    const remote = connection?.mode === 'remote'

    const version = appVersion ? `v${appVersion}` : (sha ?? copy.unknown)
    const base = remote ? copy.clientLabel(appVersion ?? sha ?? copy.unknown) : version
    const behindHint = !applying && behind > 0 ? ` (+${behind})` : ''

    const label = applying
      ? `${base} · ${updateApply.stage === 'restart' ? copy.restart : copy.update}`
      : `${base}${behindHint}`

    const tooltip = [
      applying ? updateApply.message || copy.updateInProgress : null,
      !applying && behind > 0 && copy.commitsBehind(behind, updateStatus?.branch ?? '...'),
      appVersion && copy.desktopVersion(appVersion),
      sha && copy.commit(sha),
      updateStatus?.branch && copy.branch(updateStatus.branch)
    ]
      .filter(Boolean)
      .join(' · ')

    return {
      className: !applying && behind > 0 ? 'text-primary hover:text-primary' : undefined,
      detail: appVersion && sha && !applying && !remote ? sha : undefined,
      hidden: !appVersion && !sha,
      icon: applying ? <Loader2 className="size-3 animate-spin" /> : <Hash className="size-3" />,
      id: 'version-client',
      label,
      onSelect: () => openUpdateOverlayFor('client'),
      title: tooltip || undefined,
      variant: 'action'
    }
  }, [
    desktopVersion?.appVersion,
    connection?.mode,
    copy,
    updateApply.applying,
    updateApply.message,
    updateApply.stage,
    updateStatus?.behind,
    updateStatus?.branch,
    updateStatus?.currentSha
  ])

  const backendVersionItem = useMemo<StatusbarItem | null>(() => {
    if (connection?.mode !== 'remote') {
      return null
    }

    const backendVersion = statusSnapshot?.version
    const behind = backendUpdateStatus?.behind ?? 0
    const applying = backendUpdateApply.applying || backendUpdateApply.stage === 'restart'

    const base = copy.backendLabel(backendVersion ?? copy.unknown)
    const behindHint = !applying && behind > 0 ? ` (+${behind})` : ''

    const label = applying
      ? `${base} · ${backendUpdateApply.stage === 'restart' ? copy.restart : copy.update}`
      : `${base}${behindHint}`

    const tooltip = [
      applying ? backendUpdateApply.message || copy.updateInProgress : null,
      !applying && behind > 0 && copy.commitsBehind(behind, 'main'),
      backendVersion && copy.backendVersion(backendVersion)
    ]
      .filter(Boolean)
      .join(' · ')

    return {
      className: !applying && behind > 0 ? 'text-primary hover:text-primary' : undefined,
      hidden: !backendVersion,
      icon: applying ? <Loader2 className="size-3 animate-spin" /> : <Hash className="size-3" />,
      id: 'version-backend',
      label,
      onSelect: () => openUpdateOverlayFor('backend'),
      title: tooltip || undefined,
      variant: 'action'
    }
  }, [
    connection?.mode,
    statusSnapshot?.version,
    backendUpdateStatus?.behind,
    backendUpdateApply.applying,
    backendUpdateApply.message,
    backendUpdateApply.stage,
    copy
  ])

  // Connection-identity pill (VS Code's load-bearing "where am I?" cue). Shown
  // only for remote connections; hidden in local mode (the unmarked default).
  // SSH remotes read "SSH: user@host"; token/oauth remotes read "Remote: host"
  // — closing the same gap for the existing remote modes. Clicking opens the
  // gateway connection settings so the pill doubles as the switch/disconnect
  // entry point.
  const connectionItem = useMemo<StatusbarItem | null>(() => {
    if (connection?.mode !== 'remote') {
      return null
    }
    // Prefer the host main.cjs put on the descriptor; fall back to parsing the
    // backend URL (never the 127.0.0.1 tunnel — that's only the SSH baseUrl,
    // and SSH descriptors always carry remoteHost).
    let host = connection.remoteHost ?? ''
    if (!host && connection.baseUrl) {
      try {
        host = new URL(connection.baseUrl).host
      } catch {
        host = ''
      }
    }
    if (!host) {
      return null
    }

    const isSsh = connection.remoteKind === 'ssh'
    const label = isSsh ? copy.connectionSsh(host) : copy.connectionRemote(host)
    const baseTooltip = isSsh ? copy.connectionSshTooltip(host) : copy.connectionRemoteTooltip(host)
    // Append the per-profile scope when this is a profile-scoped connection, so
    // the pill discloses WHICH profile the host backs (not just the host).
    const profile = connection.profile
    const title = profile ? `${baseTooltip} · ${profile}` : baseTooltip

    return {
      // VS Code-style remote indicator: a solid colored block (not a muted
      // pill) so "you are running on a remote host" is unmistakable, pinned to
      // the FAR LEFT of the status bar. SSH gets the primary accent; a plain URL
      // remote gets a calmer tint so the two are visually distinct.
      className: cn(
        'px-2 font-medium',
        isSsh
          ? 'bg-primary text-primary-foreground hover:bg-primary/90 hover:text-primary-foreground'
          : 'bg-accent text-accent-foreground hover:bg-accent/90 hover:text-accent-foreground'
      ),
      icon: <Network className="size-3" />,
      id: 'connection',
      label,
      title,
      // Deep-link straight to the Gateway connection panel (the settings index
      // reads ?tab=), so the pill lands the user where they manage/switch it.
      // NB: default (button) variant — NOT 'link', which renders an <a href> and
      // would swallow the in-app `to:` navigation.
      to: `${SETTINGS_ROUTE}?tab=gateway`
    }
  }, [connection?.mode, connection?.remoteHost, connection?.remoteKind, connection?.baseUrl, connection?.profile, copy])

  const coreLeftStatusbarItems = useMemo<readonly StatusbarItem[]>(
    () => [
      // Remote-connection indicator pinned to the far left (VS Code parity) —
      // first thing in the bar so "where am I running" is the dominant cue.
      // Absent in local mode.
      ...(connectionItem ? [connectionItem] : []),
      {
        className: `w-7 justify-center px-0${commandCenterOpen ? ' bg-accent/55 text-foreground' : ''}`,
        icon: <Command className="size-3.5" />,
        id: 'command-center',
        onSelect: toggleCommandCenter,
        title: commandCenterOpen ? copy.closeCommandCenter : copy.openCommandCenter,
        variant: 'action'
      },
      {
        className: gatewayRestarting ? undefined : gatewayClassName,
        detail: gatewayRestarting ? copy.gatewayRestarting : gatewayDetail,
        icon: gatewayRestarting ? (
          <GlyphSpinner ariaLabel={copy.gatewayRestarting} className="size-3" />
        ) : inferenceReady ? (
          <Activity className="size-3" />
        ) : (
          <AlertCircle className="size-3" />
        ),
        id: 'gateway-health',
        label: copy.gateway,
        menuClassName: 'w-72',
        menuContent: gatewayMenuContent,
        title: inferenceStatus?.reason || copy.gatewayTitle,
        variant: 'menu'
      },
      {
        className: cn(
          agentsOpen && 'bg-accent/55 text-foreground',
          bgFailed > 0 && 'text-destructive hover:text-destructive'
        ),
        detail:
          subagentsRunning > 0
            ? copy.subagents(subagentsRunning)
            : bgFailed > 0
              ? copy.failed(bgFailed)
              : bgRunning > 0
                ? copy.running(bgRunning)
                : undefined,
        icon:
          bgFailed > 0 ? (
            <AlertCircle className="size-3" />
          ) : bgRunning > 0 || subagentsRunning > 0 ? (
            <Loader2 className="size-3 animate-spin" />
          ) : (
            <Sparkles className="size-3" />
          ),
        id: 'agents',
        label: copy.agents,
        onSelect: openAgents,
        title: agentsOpen ? copy.closeAgents : copy.openAgents,
        variant: 'action'
      },
      {
        icon: <Clock className="size-3" />,
        id: 'cron',
        label: copy.cron,
        title: copy.openCron,
        to: CRON_ROUTE,
        variant: 'action'
      }
    ],
    [
      agentsOpen,
      bgFailed,
      bgRunning,
      commandCenterOpen,
      connectionItem,
      copy,
      gatewayMenuContent,
      gatewayClassName,
      gatewayDetail,
      gatewayRestarting,
      inferenceReady,
      inferenceStatus?.reason,
      openAgents,
      subagentsRunning,
      toggleCommandCenter
    ]
  )

  const coreRightStatusbarItems = useMemo<readonly StatusbarItem[]>(
    () => [
      {
        detail: <LiveDuration since={turnStartedAt} />,
        hidden: !busy || !turnStartedAt,
        icon: <Loader2 className="size-3 animate-spin" />,
        id: 'running-timer',
        label: copy.turnRunning,
        title: copy.currentTurnElapsed,
        variant: 'text'
      },
      {
        detail: contextBar || undefined,
        hidden: !contextUsage,
        id: 'context-usage',
        label: contextUsage,
        title: copy.contextUsage,
        variant: 'text'
      },
      {
        detail: <LiveDuration since={sessionStartedAt} />,
        hidden: !sessionStartedAt,
        id: 'session-timer',
        label: copy.session,
        title: copy.runtimeSessionElapsed,
        variant: 'text'
      },
      {
        className: cn('px-1', yoloActive && 'bg-(--chrome-action-hover)'),
        hidden: !showYoloToggle,
        icon: yoloActive ? (
          <ZapFilled className="size-3.5 shrink-0" />
        ) : (
          <Zap className="size-3.5 shrink-0 opacity-70" />
        ),
        id: 'yolo',
        onSelect: modifiers => void toggleYolo(modifiers),
        title: yoloActive ? copy.yoloOn : copy.yoloOff,
        variant: 'action'
      },
      {
        className: `w-7 justify-center px-0${terminalTakeover ? ' bg-accent/55 text-foreground' : ''}`,
        hidden: !chatOpen,
        icon: <Terminal className="size-3.5" />,
        id: 'terminal',
        onSelect: () => setTerminalTakeover(!$terminalTakeover.get()),
        title: terminalTakeover ? copy.hideTerminal : copy.showTerminal,
        variant: 'action'
      },
      clientVersionItem,
      ...(backendVersionItem ? [backendVersionItem] : [])
    ],
    [
      busy,
      chatOpen,
      contextBar,
      contextUsage,
      copy,
      sessionStartedAt,
      showYoloToggle,
      terminalTakeover,
      toggleYolo,
      turnStartedAt,
      clientVersionItem,
      backendVersionItem,
      yoloActive
    ]
  )

  const leftStatusbarItems = useMemo(
    () => [...coreLeftStatusbarItems, ...extraLeftItems],
    [coreLeftStatusbarItems, extraLeftItems]
  )

  const statusbarItems = useMemo(
    () => [...extraRightItems, ...coreRightStatusbarItems],
    [coreRightStatusbarItems, extraRightItems]
  )

  return { leftStatusbarItems, statusbarItems }
}
