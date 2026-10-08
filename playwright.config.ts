import { defineConfig } from '@playwright/test';

const baseURL = process.env.TENSORLAB_TEST_URL || 'http://127.0.0.1:8765';

export default defineConfig({
  testDir: './e2e', timeout: 90000, workers: 1,
  use: { baseURL, headless: true, channel: 'msedge', screenshot: 'only-on-failure', trace: 'retain-on-failure', viewport: { width: 1440, height: 900 }, launchOptions: { args: ['--enable-webgl', '--ignore-gpu-blocklist'] } },
  webServer: process.env.CI ? {
    command: `".venv/Scripts/python.exe" run.py --no-browser --port ${new URL(baseURL).port || '8765'}`,
    url: `${baseURL}/api/health`,
    timeout: 120000,
    reuseExistingServer: false,
    stdout: 'pipe',
    stderr: 'pipe'
  } : undefined,
  reporter: 'list'
});
