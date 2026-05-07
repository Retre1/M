# TradingView → OKX setup гайд

> Пошаговый запуск: Pine Script стратегия → твой webhook сервер → OKX исполнение
> Время setup: ~2 часа (Telegram bot 5 мин + OKX KYC 1-2 дня + VPS deploy 30 мин + TV alerts 15 мин)

---

## Архитектура

```
┌──────────────────────────────────────────────────────────────┐
│  TradingView (Premium / Pro+)                                 │
│  Pine Script strategy: ApexFX Donchian Turtle                 │
│  Bar close → emit JSON alert                                  │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTPS POST (webhook)
                         ↓
┌──────────────────────────────────────────────────────────────┐
│  Your VPS ($5/month — Hetzner / DigitalOcean / Vultr)        │
│  Flask webhook receiver на порту 8080                         │
│  → HMAC verify → pydantic validate → dedup → handler          │
└────────────────────────┬─────────────────────────────────────┘
                         │ OKX REST API
                         ↓
┌──────────────────────────────────────────────────────────────┐
│  OKX Demo Trading (testnet) → 30 дней paper                   │
│  Затем: OKX live USDT-M perpetual                            │
└──────────────────────────────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────┐
│  Telegram bot — alerts на всё                                 │
│  (entry, pyramid, exit, kill switch, daily summary)           │
└──────────────────────────────────────────────────────────────┘
```

---

## Шаг 1 — TradingView подписка

Нужен **минимум Pro+** ($30/мес) для webhook alerts.  Free / Pro дают только email/popup alerts — нам они не подходят, бот не может прочитать почту в realtime.

| План | Цена | Webhook? | Кол-во alerts |
|---|---|---|---|
| Basic | бесплатно | ❌ | 1 |
| Essential | $13/мес | ❌ | 20 |
| Plus | $25/мес | ❌ | 100 |
| **Pro+** | **$30/мес** | **✅** | **400** |
| Premium | $60/мес | ✅ | 1000 |

Для нашего случая **Pro+ достаточно** (3 пары × 4 alerts на стратегию = ~12 alerts постоянно активных, далеко от лимита 400).

Если у тебя есть Pro+ — продолжай.  Если нет — сэкономь на этом не получится, webhook это требование для автоматизации.

---

## Шаг 2 — OKX регистрация + KYC

```
1. okx.com → Sign Up (используй ProtonMail / Tutanota — анонимнее)
2. 2FA через Google Authenticator (НЕ SMS)
3. Identity Verification → Standard
   - Загрузи скан российского паспорта
   - Selfie с паспортом
   - Wait 1-3 рабочих дня для approval
```

После approval:

```
4. Demo Trading Mode (вверху сайта переключатель: Demo / Real)
   - В Demo получишь $100,000 фейковых USDT
   - Используем сначала ТОЛЬКО demo для paper trading
5. API Management → Create API Key
   - Permissions: ✅ Read, ✅ Trade, ❌ Withdraw
   - IP whitelist: твой VPS IP (получим в шаге 4)
   - Сохрани: API Key, API Secret, Passphrase (3 значения!)
```

⚠️ **Демо-режим включается через `x-simulated-trading: 1` header** — наш Python клиент делает это автоматически когда `OkxClient(demo=True)`.

---

## Шаг 3 — Telegram bot (5 минут)

```
1. Открой Telegram → найди @BotFather
2. /newbot → выбери имя (например, ApexFX Trader)
3. Сохрани токен (формат: 1234567890:AAH...)
4. Создай chat: напиши боту любое сообщение
5. Открой в браузере:
   https://api.telegram.org/bot<ТВОЙ_ТОКЕН>/getUpdates
6. Найди в JSON: result[0].message.chat.id (число)
   - Для приватного чата это положительное число
   - Для группы — отрицательное (если хочешь алерты в группу)
```

Проверь что бот отвечает:
```bash
curl -X POST "https://api.telegram.org/bot<ТОКЕН>/sendMessage" \
  -d "chat_id=<CHAT_ID>" -d "text=test from terminal"
```

---

## Шаг 4 — VPS деплой webhook receiver

### 4.1. Создать VPS

Любой провайдер — **Hetzner CX11** ($4/мес), **DigitalOcean Droplet** ($6/мес), **Vultr** ($6/мес) подойдут. Россияне обычно используют:

- **Hetzner** (Германия) — оплата картой, можно через крипту
- **AEZA** (РФ) — без VPN, рублёвая оплата
- **timeweb cloud** (РФ) — рубли

Минимум: 1 vCPU, 1GB RAM, Ubuntu 22.04. **Запиши IP — он нужен для OKX whitelist (Шаг 2)**.

### 4.2. Setup на VPS

```bash
ssh root@<VPS_IP>

# Базовые пакеты
apt update && apt install -y python3.11 python3.11-venv git nginx certbot python3-certbot-nginx

# Клон проекта
git clone https://github.com/Retre1/M.git apexfx
cd apexfx
git checkout main
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install gunicorn

# Env файл
cat > /etc/apexfx.env <<'EOF'
APEXFX_WEBHOOK_SECRET=замени-на-длинную-случайную-строку-минимум-32-символа
APEXFX_TELEGRAM_TOKEN=твой-telegram-bot-token
APEXFX_TELEGRAM_CHAT_ID=твой-chat-id
APEXFX_OKX_API_KEY=твой-okx-api-key
APEXFX_OKX_API_SECRET=твой-okx-secret
APEXFX_OKX_API_PASSPHRASE=твой-okx-passphrase
APEXFX_OKX_DEMO=true
EOF
chmod 600 /etc/apexfx.env

# Сгенерируй webhook secret
openssl rand -hex 32  # → используй вывод как APEXFX_WEBHOOK_SECRET
```

### 4.3. systemd сервис

```bash
cat > /etc/systemd/system/apexfx-webhook.service <<'EOF'
[Unit]
Description=ApexFX Webhook Receiver
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/apexfx
EnvironmentFile=/etc/apexfx.env
ExecStart=/root/apexfx/.venv/bin/gunicorn \
  -w 1 -b 127.0.0.1:8080 \
  --timeout 30 --keep-alive 5 --log-level info \
  'apexfx.aggressive.webhook.server:create_app_from_env()'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now apexfx-webhook
systemctl status apexfx-webhook  # должно быть active (running)
```

### 4.4. Nginx + SSL (TradingView требует HTTPS)

```bash
# Получи домен (Cloudflare свой бесплатно даёт через workers, или namecheap)
# Прицепи A-запись к твоему VPS IP

cat > /etc/nginx/sites-available/apexfx <<'EOF'
server {
    listen 80;
    server_name your-domain.com;
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 30s;
    }
}
EOF

ln -s /etc/nginx/sites-available/apexfx /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# SSL через Let's Encrypt (бесплатно)
certbot --nginx -d your-domain.com --non-interactive --agree-tos -m you@example.com
```

Проверь health endpoint:
```bash
curl https://your-domain.com/health
# Ожидаешь: {"status":"ok","service":"apexfx-webhook",...}
```

---

## Шаг 5 — Pine Script в TradingView

```
1. Открой TradingView → BTCUSDT.P (или любой другой crypto perp)
   Источник данных: OKX (вверху графика)
2. Timeframe: 4H
3. Pine Editor (внизу) → New → Strategy
4. Удали стартовый код, вставь ПОЛНОСТЬЮ файл:
   /Users/abobik/Desktop/M/src/apexfx/aggressive/tradingview/donchian_turtle.pine
5. Save (Ctrl+S) → имя "ApexFX Donchian Turtle"
6. Click "Add to chart" — стратегия отрисуется
```

### 5.1. Backtest

В нижней панели появится **Strategy Tester**.  Targets для запуска live:

| Метрика | Цель |
|---|---|
| Net profit (1+ год) | > +50% |
| Profit factor | > 1.3 |
| Max drawdown | < 35% |
| Sharpe ratio | > 1.0 |
| Total trades | > 30 (не overfit) |

Если на BTCUSDT 4H 2022-2026 не достигаешь — **не запускай live**, тюнингуй параметры:
- Поэксперементируй с `entry_period` (15, 20, 30)
- Включи/выключи `use_trend_filter`
- Поменяй `risk_per_unit_pct` (1% → 2%)

### 5.2. Multi-symbol verification

Применил ТОТ ЖЕ Pine Script на:
- ETHUSDT.P
- SOLUSDT.P

**Стратегия должна быть прибыльной на ВСЕХ ТРЁХ.** Если на одной — нет, она overfit под конкретную пару.

### 5.3. Setup alerts (для live trading)

```
1. Right-click on chart → Add alert
2. Condition:
   Symbol/Strategy: "ApexFX Donchian Turtle"
   Trigger: "Any alert() function call"
3. Actions:
   ✅ Webhook URL
   URL: https://your-domain.com/tv-webhook
4. Notifications:
   ✅ Show pop-up (для дебага)
   ✅ Send email (резервный канал)
5. Message: ОСТАВЬ ПУСТЫМ
   (Pine Script сам формирует JSON через alert_message)
6. Frequency: ⭕ "Once per bar close"
   (НЕ "Once per bar" — это даст intra-bar repaints)
7. Webhook headers:
   X-Webhook-Secret: <твой-секрет-из-/etc/apexfx.env>
8. Click Create
```

⚠️ **Headers:** TradingView Pro+ позволяет custom headers в alerts. Без них наш сервер вернёт 401. Если на твоём плане headers недоступны — апгрейдь до Premium.

### 5.4. Test alert

```
1. На графике: правый клик на свечу → "Trigger Alert" (если уже создан)
2. Проверь логи на VPS:
   journalctl -u apexfx-webhook -n 50
   - Должно быть либо "accepted" либо понятная ошибка
3. Проверь Telegram — должен прийти message о signal
```

---

## Шаг 6 — 30 дней Paper trading

Этот шаг **ОБЯЗАТЕЛЕН**.

```
Days 1-7: Watch & log
   - Bot работает в demo mode
   - Каждое утро смотри: trades за вчера, PnL, есть ли ошибки в логах
   - Telegram alerts — все приходят?

Days 8-14: Reconciliation
   - Сравни paper Telegram alerts с TradingView Strategy Tester за тот же период
   - Расхождение должно быть < 5% по PnL (учитывая slippage и spread)

Days 15-21: Stress test
   - Создай fake "market crash" в Pine: добавь временные input для теста
   - Проверь kill switch активируется как ожидалось
   - Проверь auto re-arm cooldown работает

Days 22-30: Final validation
   - Если PF > 1.0, Sharpe > 1.0, no system errors —
     → готов к micro-live $200
   - Если есть проблемы — fix, restart 30 дней с нуля
```

---

## Шаг 7 — Micro-live ($200 → $1000)

```
Week 1 (micro-live $200):
1. Переключи OKX_DEMO=false в /etc/apexfx.env
2. systemctl restart apexfx-webhook
3. Депозит $200 USDT на OKX (через P2P с Тинькофф/Сбер)
4. Risk per unit: 0.25% ($0.50)
   (в Pine: Settings → risk_per_unit_pct = 0.25)
5. Только наблюдение, ZERO override

Week 2-4: Если выжил — продолжай
   - Месячный ревью: PF, max DD, total trades
   - Если все в плюс → депозит до $500

Week 5-8: Scale to $1000
   - Risk per unit обратно 1.5%
   - Полный capital deployment
```

---

## Troubleshooting

### Webhook возвращает 401
- Проверь header `X-Webhook-Secret` в TV alert — точно совпадает с `APEXFX_WEBHOOK_SECRET` на сервере?
- Проверь secret не имеет лишних пробелов / переносов строк
- `journalctl -u apexfx-webhook | grep "auth failed"` — покажет неудачные попытки

### TV alert не доходит
- Pro+ план активен?
- Webhook URL — HTTPS, не HTTP
- Domain резолвится? `dig your-domain.com`
- Curl с твоего ноутбука: `curl https://your-domain.com/health`
- TradingView сам логирует webhook delivery — Settings → Notifications → History

### OKX rejects orders
- API key activated? (после создания иногда нужно подождать 30 мин)
- Permissions включают Trade?
- IP whitelist содержит твой VPS IP?
- Demo mode правильно установлен? (`x-simulated-trading: 1` header добавляется автоматически)

### Bot молчит, но в логах "accepted"
- Telegram bot blocked? Проверь chat_id правильный
- Token не expired?
- `curl https://api.telegram.org/bot<TOKEN>/sendMessage -d "chat_id=<ID>&text=test"`

---

## Стоимость per month

| Компонент | Цена |
|---|---|
| TradingView Pro+ | $30 |
| VPS (Hetzner CX11) | $4 |
| Domain (Cloudflare) | $0 (или $10/год через namecheap) |
| SSL (Let's Encrypt) | $0 |
| Telegram | $0 |
| OKX trading fees (~100 trades/мес) | ~$5-10 |
| **Total** | **~$40/мес** |

На $1k капитала это 4% от баланса — но это **fixed cost**.  Когда счёт растёт до $5k+, то это уже < 1%.

---

## Чеклист готовности к live

Перед переключением `OKX_DEMO=false`:

- [ ] 30 дней paper trading с PF > 1.0
- [ ] Backtest BTCUSDT/ETHUSDT/SOLUSDT — все в плюс на 2+ годах
- [ ] Webhook auth работает (тест-alert принят с правильным secret, отвергнут без)
- [ ] Все Telegram alert templates показывают ожидаемый формат
- [ ] Kill switch вручную проверен (`touch .kill_switch` → бот остановился)
- [ ] Auto-trip kill switch проверен (manually drop equity below threshold)
- [ ] Daily summary доходит в полночь UTC
- [ ] OKX API IP whitelist setup
- [ ] OKX API ключи без Withdraw permission
- [ ] /etc/apexfx.env права 600 (только root читает)
- [ ] Nginx SSL валидный (`certbot certificates`)
- [ ] systemd auto-restart включён (`systemctl is-enabled apexfx-webhook`)

Если хоть один пункт не выполнен — не идти в live.

---

*Этот гайд предполагает Pro+ TradingView подписку и retail OKX аккаунт.  Для других setup'ов (Bybit, Hyperliquid) основные шаги те же, меняется только exchange client (`okx_client.py` → `bybit_client.py`).  Pine Script универсален.*
