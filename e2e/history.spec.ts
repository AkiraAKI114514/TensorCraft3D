import { test, expect } from '@playwright/test';

test('supports keyboard undo and redo for graph edits', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('模型模板').selectOption('mlp');
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator('.react-flow__node[data-id="layer_1"]').click();
  const width = page.getByLabel('out_features', { exact: true });
  await expect(width).toHaveValue('64');

  await width.fill('12');
  await width.blur();
  await expect(width).toHaveValue('12');
  await page.keyboard.press('Control+z');
  await expect(width).toHaveValue('64');
  await page.keyboard.press('Control+Shift+z');
  await expect(width).toHaveValue('12');
  await page.keyboard.press('Control+z');
  await expect(width).toHaveValue('64');
  await page.keyboard.press('Control+y');
  await expect(width).toHaveValue('12');
});
