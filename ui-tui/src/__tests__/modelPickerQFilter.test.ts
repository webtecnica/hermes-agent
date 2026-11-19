import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import type { ModelOptionProvider } from '@hermes/shared/gateway-events'
import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import { ModelPicker } from '../components/modelPicker.js'
import { DEFAULT_THEME } from '../theme.js'

// #157542: with an empty filter the first `q` used to close the picker instead
// of becoming the first filter character, so `qwen*` was unreachable through
// type-to-filter. On the list stages (provider, model) a printable key belongs
// to the filter; only Esc backs out.

const OPENROUTER: ModelOptionProvider = {
  is_current: true,
  models: ['gpt-4o', 'glm-4.6'],
  name: 'OpenRouter',
  slug: 'openrouter',
  total_models: 2
}

const QWEN: ModelOptionProvider = {
  models: ['qwen3-max', 'qwen3-coder', 'deepseek-v3'],
  name: 'Qwen (DashScope)',
  slug: 'dashscope',
  total_models: 3
}

const pending = () => new Promise(resolve => setImmediate(resolve))

function mount(providers: ModelOptionProvider[], model = 'gpt-4o') {
  const request = vi.fn(async (method: string) => {
    if (method !== 'model.options') {
      throw new Error(`unexpected RPC ${method}`)
    }

    return { model, providers }
  })

  const stdin = Object.assign(new PassThrough(), {
    isTTY: true,
    ref: vi.fn(),
    setRawMode: vi.fn(),
    unref: vi.fn()
  })

  const stdout = Object.assign(new PassThrough(), { columns: 100, isTTY: false, rows: 40 })
  const stderr = new PassThrough()
  const onCancel = vi.fn()

  // Only the frame produced by the current keypress matters: older frames stay
  // in the accumulated stream and would defeat the negative assertions.
  const frame = { text: '' }

  stdout.on('data', chunk => {
    frame.text += chunk.toString()
  })

  const instance = renderSync(
    React.createElement(ModelPicker, {
      gw: { request } as any,
      onCancel,
      onSelect: vi.fn(),
      sessionId: null,
      t: DEFAULT_THEME
    }),
    { patchConsole: false, stderr: stderr as any, stdin: stdin as any, stdout: stdout as any }
  )

  /** Press a key sequence and return the (ANSI-stripped) frame it produced. */
  const press = async (input: string) => {
    frame.text = ''
    stdin.write(input)
    // The fork re-renders on its own frame throttle: give React's commit and
    // the frame write a real timer to land before reading the diff.
    await new Promise(resolve => setTimeout(resolve, 80))
    await pending()

    return stripAnsi(frame.text)
  }

  return {
    onCancel,
    press,
    cleanup: () => {
      instance.unmount()
      instance.cleanup()
    }
  }
}

describe('ModelPicker type-to-filter on q (#157542)', () => {
  it('types q into the provider filter instead of closing the picker', async () => {
    const picker = mount([OPENROUTER, QWEN])

    try {
      expect(await picker.press('')).toContain('OpenRouter')

      // First character of an empty filter: a filter character, not a cancel.
      const typed = await picker.press('q')

      expect(picker.onCancel).not.toHaveBeenCalled()
      expect(typed).toContain('Qwen')
      expect(typed).not.toContain('OpenRouter')
      expect(typed).toContain('filter: q')
    } finally {
      picker.cleanup()
    }
  })

  it('types q into the model filter so qwen* models stay reachable', async () => {
    const picker = mount([OPENROUTER, QWEN])

    try {
      // Wait for the initial model.options frame before pressing anything.
      expect(await picker.press('')).toContain('OpenRouter')

      // provider stage: filter down to Qwen, then Enter moves to the model stage
      await picker.press('q')
      const models = await picker.press('\r')

      expect(models).toContain('qwen3-max')
      expect(models).toContain('deepseek-v3')

      const typed = await picker.press('q')

      expect(picker.onCancel).not.toHaveBeenCalled()
      expect(typed).toContain('qwen3-max')
      expect(typed).toContain('qwen3-coder')
      expect(typed).not.toContain('deepseek-v3')
      expect(typed).toContain('filter: q')
    } finally {
      picker.cleanup()
    }
  })

  it('still backs out with Esc on the provider stage, where q no longer closes', async () => {
    const picker = mount([OPENROUTER, QWEN])

    try {
      expect(await picker.press('')).toContain('OpenRouter')

      // `q` is a filter character now, so Esc is the way out of the list stages.
      await picker.press('\u001b')
      await vi.waitFor(() => expect(picker.onCancel).toHaveBeenCalledTimes(1))
    } finally {
      picker.cleanup()
    }
  })

  it('still closes on q from the effort stage, which has no filter', async () => {
    const picker = mount([OPENROUTER, QWEN])

    try {
      expect(await picker.press('')).toContain('OpenRouter')

      // provider → model → effort (step 3/3): Enter twice.
      await picker.press('\r')
      const effort = await picker.press('\r')

      expect(effort).toContain('Reasoning effort')

      await picker.press('q')

      expect(picker.onCancel).toHaveBeenCalledTimes(1)
    } finally {
      picker.cleanup()
    }
  })

  it('still closes on q from the API-key stage, which is not a list stage', async () => {
    const unconfigured: ModelOptionProvider = {
      auth_type: 'api_key',
      authenticated: false,
      key_env: 'DASHSCOPE_API_KEY',
      models: [],
      name: 'Qwen (DashScope)',
      slug: 'dashscope',
      total_models: 0
    }

    const picker = mount([unconfigured])

    try {
      expect(await picker.press('')).toContain('Qwen')

      const keyStage = await picker.press('\r') // Enter on the unauthenticated provider → key stage

      expect(keyStage).toContain('DASHSCOPE_API_KEY')

      // Off the list stages the shared overlay handler still owns `q`.
      await picker.press('q')

      expect(picker.onCancel).toHaveBeenCalledTimes(1)
    } finally {
      picker.cleanup()
    }
  })
})
