import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { PRESETS } from '../src/presets';

const savedProject = JSON.stringify(PRESETS.mlp());

test('backs up a saved project after a React error and recovers without deleting it', async ({ page }, testInfo) => {
  await page.addInitScript(saved => {
    if (!sessionStorage.getItem('tensorlab-test-initialized')) {
      localStorage.setItem('tensorlab-project', saved);
      localStorage.setItem('tensorlab-unrelated', 'keep');
      sessionStorage.setItem('tensorlab-test-initialized', 'yes');
      sessionStorage.setItem('tensorlab-test-crash', 'yes');
    }
    const toFixed = Number.prototype.toFixed;
    Number.prototype.toFixed = function(digits) {
      if (digits === 2 && sessionStorage.getItem('tensorlab-test-crash')) throw new Error('测试工作台渲染异常');
      return toFixed.call(this, digits);
    };
  }, savedProject);
  await page.goto('/');
  const fallback = page.getByRole('alert');
  await expect(fallback).toContainText('工作台发生错误');
  await expect(page.locator('.app-shell')).toHaveCount(0);
  expect(await page.evaluate(() => localStorage.getItem('tensorlab-project'))).toBe(savedProject);
  await fallback.getByText('错误详情', { exact: true }).click();
  await expect(fallback.locator('pre')).toHaveText('测试工作台渲染异常');
  const pending = page.waitForEvent('download');
  await fallback.getByRole('button', { name: '下载已保存项目', exact: true }).click();
  const backup = await pending;
  expect(backup.suggestedFilename()).toBe('tensorcraft3d-recovery.json');
  expect(readFileSync((await backup.path())!, 'utf8')).toBe(savedProject);
  await expect(fallback.getByRole('status')).toContainText('最近一次成功自动保存');
  await page.screenshot({ path: testInfo.outputPath('recovery-desktop.png') });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('recovery-mobile.png') });
  await Promise.all([
    page.waitForEvent('load'),
    fallback.getByRole('button', { name: '重新加载工作台', exact: true }).click(),
  ]);
  await expect(page.getByRole('alert')).toContainText('工作台发生错误');
  expect(await page.evaluate(() => localStorage.getItem('tensorlab-project'))).toBe(savedProject);
  await page.evaluate(() => sessionStorage.removeItem('tensorlab-test-crash'));
  await page.getByRole('button', { name: '重新加载工作台', exact: true }).click();
  await expect(page.locator('.app-shell')).toBeVisible();
  await expect(page.locator('.app-error-screen')).toHaveCount(0);
  expect(await page.evaluate(() => localStorage.getItem('tensorlab-project'))).toBe(savedProject);
  expect(await page.evaluate(() => localStorage.getItem('tensorlab-unrelated'))).toBe('keep');
});

test('keeps recovery actions usable when browser storage is unavailable', async ({ page }) => {
  await page.addInitScript(() => {
    const getItem = Storage.prototype.getItem, setItem = Storage.prototype.setItem;
    Storage.prototype.getItem = function(key) {
      if (this === localStorage && key === 'tensorlab-project') throw new Error('浏览器存储不可用');
      return getItem.call(this, key);
    };
    Storage.prototype.setItem = function(key, value) {
      if (this === localStorage && key === 'tensorlab-project') throw new Error('浏览器存储不可用');
      setItem.call(this, key, value);
    };
  });
  await page.goto('/');
  const fallback = page.getByRole('alert');
  await expect(fallback).toContainText('工作台发生错误');
  await fallback.getByRole('button', { name: '下载已保存项目', exact: true }).click();
  await expect(fallback.getByRole('status')).toContainText('无法读取或下载');
  await expect(fallback.getByRole('button', { name: '重新加载工作台', exact: true })).toBeEnabled();
});

test('keeps an editable graph with shape errors in the normal workbench', async ({ page }) => {
  const graph = PRESETS.mlp(); graph.nodes[1].params.out_features = -1;
  await page.addInitScript(saved => localStorage.setItem('tensorlab-project', saved), JSON.stringify(graph));
  await page.goto('/');
  await expect(page.locator('.app-shell')).toBeVisible();
  await expect(page.locator('.validation-badge')).toContainText('结构错误');
  await expect(page.locator('.app-error-screen')).toHaveCount(0);
  await page.getByRole('button', { name: '导出代码', exact: true }).click();
  await expect(page.getByRole('button', { name: '下载 .py', exact: true })).toBeDisabled();
});
