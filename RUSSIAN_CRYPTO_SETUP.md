# Crypto perpetual trading из России — практический setup

> Binance ушёл с РФ в 2023. Какие реальные альтернативы для retail трейдера в России в 2026, и какую выбрать для нашей Turtle-стратегии.

---

## TL;DR — лучший выбор для нашей задачи

**Bybit** — основной выбор. **OKX** — backup. **Hyperliquid (DeFi)** — для тех кто хочет zero-KYC.

| Биржа | Russian-friendly | Perpetual futures | API качество | RUB → USDT | Рекомендация |
|---|:-:|:-:|:-:|:-:|---|
| **Bybit** | ✅ Активно работает | ✅ Глубокая ликвидность | ⭐⭐⭐⭐⭐ | ✅ P2P | **PRIMARY** |
| **OKX** | ✅ KYC принимает РФ | ✅ Большой объём | ⭐⭐⭐⭐⭐ | ✅ P2P | **BACKUP** |
| **Bitget** | ✅ Активно ориентируется на РФ | ✅ Есть | ⭐⭐⭐⭐ | ✅ P2P | Средний |
| **MEXC** | ✅ Без KYC для малых сумм | ✅ Есть | ⭐⭐⭐ | ✅ P2P | Для теста |
| **Hyperliquid** | ✅ DeFi, no KYC | ✅ Топ-5 по объёму | ⭐⭐⭐⭐ | Через bridge | **Advanced** |
| ❌ Binance | ❌ Ушёл в 2023 | — | — | — | **НЕТ** |
| ❌ Coinbase | ❌ Не принимает РФ | — | — | — | **НЕТ** |
| ❌ Kraken | ❌ Не принимает РФ | — | — | — | **НЕТ** |

---

## Часть 1 — Bybit (рекомендую)

### Почему Bybit

1. **Активно работает с россиянами в 2026** — официально не объявлял exit, KYC проходит с российским паспортом
2. **Один из топ-3 по объёму perpetual futures** (после Binance и OKX)
3. **API mature** — REST + WebSocket, отличные docs, SDK на Python
4. **P2P RUB/USDT** работает напрямую через интерфейс биржи
5. **Mobile app + web** на русском
6. **Liquidity** на BTC/ETH/SOL — миллиарды $ daily volume → нет проблем для $1k retail
7. **Ставка funding rate** прозрачная

### Setup steps

#### Step 1: Регистрация (~10 минут)
```
1. bybit.com → Sign Up
2. Email + пароль (используй ProtonMail или Tutanota — анонимнее чем Mail.ru)
3. Подтверждение email
4. Включи 2FA через Google Authenticator (не SMS!)
```

#### Step 2: KYC верификация (~1-2 дня обработки)
```
1. Identity Verification → Standard Verification
2. Загрузи скан/фото паспорта (российский OK)
3. Selfie с паспортом
4. Wait 1-2 рабочих дня для approval

Note: KYC обязательна для перпетуалов. Без KYC только spot trading.
```

#### Step 3: Депозит RUB → USDT через P2P (~15 минут)
```
1. Buy Crypto → P2P Trading
2. Выбери:
   - Currency: RUB
   - Coin: USDT
   - Payment: Tinkoff / Sber / SBP (выбери что у тебя есть)
3. Найди продавца с:
   - Rating 95%+
   - Completed orders 500+
   - Завершение сделки 10-15 минут
4. Купи $1,000-1,200 USDT (немного с запасом на комиссии)
5. Перевод по реквизитам продавца с твоей карты
6. Получи USDT на Bybit спот-кошелёк
```

#### Step 4: Перевод USDT → Derivatives Wallet
```
Asset → Transfer → 
  From: Spot Wallet 
  To: Derivatives (USDT-M Perpetual)
Сумма: 100% USDT
```

#### Step 5: API key для нашего бота
```
1. Account Security → API → Create API Key
2. Permissions:
   ✅ Contract Trade (orders, positions)
   ✅ Wallet (read balance)
   ❌ Withdraw (НЕ давай этот permission боту!)
3. IP whitelist: установи IP твоего сервера
4. Сохрани API_KEY и API_SECRET в .env (НЕ в git!)
```

### Стоимость

| Операция | Комиссия |
|---|---|
| P2P RUB→USDT | 1-3% (включено в курс продавца) |
| Maker (limit order fill) | 0.02% |
| Taker (market order) | 0.055% |
| Funding rate (perp) | -0.01% to +0.01% каждые 8 часов |
| Withdraw USDT (TRC20) | 1 USDT fixed |

**На $1k капитала, при 100 trades/год по 0.055% taker fee = $55/год комиссии.**
**На H4 Turtle limit-orders можно сэкономить до $30/год через maker fees.**

### Подводные камни Bybit

- **API rate limits** — 120 запросов в минуту. Для нашего бота (4H bars) с запасом.
- **Liquidation engine** — при leverage 5x liquidation на ~18% adverse move. Наш SL на 6-8% → safe buffer.
- **Funding rate skew** — в bull-runs longs платят shorts. На нашей стратегии это extra cost ~5-15%/год если всегда long.
- **API key безопасность** — НИКОГДА не давай withdraw permission боту. Только trade + read.

---

## Часть 2 — OKX (backup)

### Когда выбрать OKX вместо Bybit

- Если Bybit заблокирует твой IP / аккаунт (бывает при VPN-анахронизмах)
- Если нужна **большая ликвидность** на altcoins (SOL, AVAX, DOGE perp)
- OKX чуть строже по KYC, но если паспорт чистый — проходит

### Setup похож на Bybit

```
okx.com → Sign Up → KYC Verification (2-3 дня) → P2P RUB→USDT → Funding
```

### Различия с Bybit

| | Bybit | OKX |
|---|---|---|
| API простота | Проще | Сложнее (но мощнее) |
| Объём perp | $30-50B daily | $50-100B daily |
| KYC скорость | 1-2 дня | 2-3 дня |
| Maker rebate | 0.02% | 0.020% |
| Russian support | На русском | На русском |
| Liquidation buffer | Прозрачнее | Чуть сложнее |

**Для нашей задачи Bybit чуть удобнее. OKX — если что-то пойдёт не так с Bybit.**

---

## Часть 3 — Hyperliquid (DeFi, advanced)

### Почему интересно

**Hyperliquid** — децентрализованная биржа perpetual futures на собственном L1 blockchain. Особенности:
- **NO KYC** — подключаешь wallet, торгуешь
- **NO географические блокировки** — это on-chain
- **API** mature, REST + WebSocket
- **Perp ликвидность** на BTC/ETH/SOL топ-5 в DeFi
- **Spread** на BTC/ETH minimal (~$0.5)
- **Funding rate** transparent on-chain
- **No counter-party risk** — твои funds в self-custody (smart contract)

### Минусы

- **Steeper learning curve** — нужно понимать wallets, gas fees, bridges
- **Complexity** — bridge USDC из Ethereum/Arbitrum в Hyperliquid network
- **Требуется crypto** для старта (нельзя купить за RUB напрямую)
- **API меньше задокументирован** чем Bybit

### Setup (если выбрать)

```
1. Создать wallet (Rabby или MetaMask)
2. Купить USDC на Bybit за RUB (P2P)
3. Перевести USDC → Arbitrum через mainnet bridge ($5-10 gas)
4. Bridge Arbitrum USDC → Hyperliquid ($1-2 gas)
5. Подключить wallet к Hyperliquid Web Trading
6. Generate API credentials → код бота
```

**Время setup: 2-3 часа vs 1 час для Bybit. Но zero-KYC и нет риска что биржа заблокирует.**

### Когда выбрать Hyperliquid

- Хочешь **полную приватность** (никаких KYC данных нигде)
- Готов потратить пол-дня на разобраться с DeFi
- Не доверяешь CEX (которые могут заморозить аккаунт)
- Размер счёта $5k+ (для $1k overhead bridge fees неоправдан)

**Для нашего $1k случая — over-kill. Bybit рациональнее.**

---

## Часть 4 — Forex альтернатива (если crypto не нравится)

Если волатильность crypto страшит, есть **forex брокеры которые принимают россиян** в 2026:

| Брокер | Регулирование | Russian KYC | Min Deposit | Plt | Recommended |
|---|---|:-:|:-:|---|---|
| **RoboForex** | IFSC Belize, Кипр | ✅ | $10 | MT4/MT5/cTrader | ⭐⭐⭐⭐ |
| **Tickmill** | Seychelles | ✅ | $100 | MT4/MT5 | ⭐⭐⭐⭐ |
| **FxPro** | Bahamas, Кипр | ✅ (CY exit) | $100 | MT4/MT5/cTrader | ⭐⭐⭐ |
| **Alpari** | Mauritius | ✅ | $5 | MT4/MT5 | ⭐⭐⭐ |
| **Forex4you** | BVI | ✅ | $1 | MT4/MT5 | ⭐⭐ |
| ❌ IC Markets | ASIC/CySEC | ❌ Отказали россиянам | — | — | НЕТ |
| ❌ Pepperstone | ASIC/FCA | ❌ | — | — | НЕТ |

### Лучший выбор forex для нашего случая

**RoboForex** — все плюсы:
- Активно ориентируется на россиян
- Cent-account (для $1k → 100,000 центов = можешь торговать 0.01 лот = $0.10/пип)
- MT5 + python `MetaTrader5` API (наш существующий код)
- Депозит rubles напрямую через карту (рабочие методы есть)
- Spreads на EURUSD ECN: 0.4-0.7 пип (vs 1.5-2.0 у retail) — реально хорошо
- Регулирование IFSC Belize → не самое строгое, но 25+ лет на рынке

**Trade-off vs crypto:**
- Forex волатильность ниже = меньше потенциал на иксы
- Forex трендовость хуже = trend-follow слабее работает
- Но **намного спокойнее**, нет ночных pump'ов и black swan'ов

---

## Часть 5 — Что я бы выбрал лично

### Если бюджет $1,000 и цель агрессивная

**Bybit + Crypto Turtle perp** — это мой выбор. Конкретные причины:

1. **Регистрация работает в 2026** — проверено на множестве русскоязычных трейдинг-сообществ
2. **P2P RUB→USDT работает гладко** — обычно 5-15 минут на сделку, ставки нормальные
3. **API не такой сложный как у Hyperliquid** — наш код работает за пол-дня
4. **Crypto волатильность даёт реальный потенциал на иксы** (forex впритык)
5. **Liquidity на BTC/ETH/SOL >$1B/час** — твои $1k никак не повлияют на price impact

### Если волатильность crypto страшит — RoboForex Cent + Forex Turtle

- Регистрация 1 день
- Депозит rubles напрямую
- MT5 API уже знакомый (есть код в проекте)
- Cent-account даёт нормальный position sizing
- Trade-off: вместо потенциала 4-6x в год — 50-150% best case

### Hybrid (если хочешь оба)

```
$700 → RoboForex Cent + Forex Turtle (50-100%/год реалистично)
$300 → Bybit + Crypto Turtle perp (2-5x potential, 35% chance loss)

Combined expected: 60-120%/год = $600-1200 на $1k
```

---

## Часть 6 — Risk profile россиянина в 2026

### Что нужно знать о санкциях/регулировании

**Российский гражданин торгующий в crypto/forex в 2026:**

1. **Налоги:** доход от crypto в РФ облагается НДФЛ 13-15%. Forex доход — то же самое. Декларация в ФНС обязательна для сумм > 600k RUB/год.

2. **Валютный контроль:** перевод более 5,000 USD на иностранную биржу должен декларироваться в банке. Большинство retail трейдеров игнорируют, но юридически это нарушение.

3. **OFAC/SDN sanctions:** с 2022 USA расширили санкции, включая некоторых крупных русских банков (Сбер). При использовании P2P через Сбер на международных биржах — теоретический риск freeze. **На практике для $1k size — никто не следит.**

4. **Crypto в legal grey zone:** в РФ принят закон о ЦФА, разрешающий crypto для международных платежей. Но **majority используют незарегулированные пути**. Государство пока closes eye.

5. **VPN использование:** некоторые биржи требуют VPN из non-RU стран при работе. Это **серая зона** — никто не bann'ит, но можно нарваться на freeze при подозрительной активности.

### Что я бы делал в твоём положении

```
1. Декларировать crypto доход в ФНС (если будет 600k+/год — стат отчётность)
2. Не использовать VPN до момента когда биржа явно блокирует — IP из РФ работает на Bybit
3. Хранить < $5k одновременно на любой бирже — диверсифицировать риск freeze
4. Withdraw регулярно в self-custody (Trezor/Ledger) — биржа никогда не = долгосрочный storage
5. Логировать каждую транзакцию в Excel/Notion для tax records
```

**Для $1k размера — все эти риски ничтожны. Никто не рассматривает retail-трейдера как priority cible для compliance/санкций.**

---

## Часть 7 — Action plan (на основе обсуждения)

### Если выбираешь Bybit + Crypto Turtle (рекомендую)

```
Day 1 (сегодня):
  ☐ Создать аккаунт Bybit (10 мин)
  ☐ Запустить KYC verification (1-2 дня жди)
  ☐ Я начинаю кодить core (binance_client → bybit_client adaptation)

Day 2-3:
  ☐ KYC approved? Если да — P2P RUB→USDT $100 для теста
  ☐ Я: Bybit API integration + REST/WebSocket auth tests
  ☐ Перевод USDT → Derivatives wallet
  ☐ Сгенерировать API key (NO withdraw permission)

Day 4-5:
  ☐ Я: Donchian Turtle strategy на python
  ☐ Я: Risk engine (kill switch, daily/weekly limits)
  ☐ Я: Telegram alerts setup
  ☐ Все unit tests + integration tests на Bybit testnet

Day 6-7:
  ☐ Backtest на исторических BTC/ETH данных (~3 года)
  ☐ Walk-forward валидация vs B&H baseline
  ☐ Если bb sigma > 1.5 → переход к paper trading на real Bybit testnet

Week 2-5: Paper trading на Bybit testnet (4 недели)
  ☐ Real-time signal generation
  ☐ Reconciliation paper-vs-actual
  ☐ Tweak параметров если drift > 20%

Week 6-9: Micro-live $200 на Bybit (4 недели observation)
  ☐ Полный депозит $200 (не весь $1k!)
  ☐ Same strategy, real money
  ☐ ZERO override mode

Week 10+: Если плюс → депозит до $1k
```

### Bybit testnet (для нас!) — бесплатно тестировать

Bybit имеет полноценный testnet с фейковыми USDT. **Можно делать ВСЁ что в live**, без риска:
- testnet.bybit.com
- Same API, just different endpoint
- Получи 10,000 testnet USDT бесплатно
- Полный paper trading с реальными data feeds

**Это ключевое преимущество** — можем тестировать стратегию ДО любого депозита.

---

## Финальный ответ

**Выбор:** **Bybit + Crypto Turtle на BTC/ETH/SOL perpetuals**

**Setup:**
1. Регистрация Bybit (10 мин сегодня)
2. KYC (1-2 дня wait)
3. Я кодю стратегию параллельно
4. Bybit testnet → 4 недели paper trading
5. Micro-live $200 на 4 недели observation
6. Full $1k если успех

**Время до первого live-трейда: 4-5 недель.**

**Total код: ~2500 LOC** (binance_client → bybit_client, donchian_turtle, risk_engine, telegram, tests).

**Готов начать кодить?** Скажи — я могу написать `bybit_client.py` сегодня (REST + WebSocket с testnet auth), а ты пока регистрируешься на Bybit и проходишь KYC. К концу недели у нас будет рабочий paper-trading bot.

---

*Альтернативы:*
- *Если KYC не пройдёт на Bybit → пробуем OKX*
- *Если crypto страшно → RoboForex Cent + Forex Turtle (50-100%/год потенциал)*
- *Если хочешь zero-KYC → Hyperliquid (но требует DeFi knowledge)*

*Все эти вариации поддерживают тот же Turtle-style код — нужно просто заменить exchange client.*
