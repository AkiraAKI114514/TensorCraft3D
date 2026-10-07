import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './e2e', timeout: 90000, workers: 1,
  use: { baseURL: process.env.TENSORLAB_TEST_URL || 'http://127.0.0.1:8765', headless: true, channel: 'msedge', screenshot: 'only-on-failure', viewport: { width: 1440, height: 900 }, launchOptions: { args: ['--enable-webgl', '--ignore-gpu-blocklist'] } },
  reporter: 'list'
});
