# Using the terminal from your phone

The web UI holds your positions and can send email from your account, so it
is built to refuse the network until it has a login:

| Setting | Listens on | Login |
|---|---|---|
| no password (default) | this machine only (`127.0.0.1`) | none |
| `python cli.py set-password` | the network (`0.0.0.0`) | required on every page |

Never forward port 8050 on your router. Each option below reaches the app
without opening anything to the internet.

On the phone, after signing in, use **Share → Add to Home Screen**: the
terminal then opens full-screen like an app, and stays signed in for 30 days.
The **🎛️ Control** tab is built for this — status at a glance, and buttons to
send today's email, run a full refresh, run AI reviews or back up the
database, each with a confirm step and a live log.

---

## Option A — your Mac + Tailscale (recommended, free, ~10 minutes)

Your data, schedule and email already run on the Mac; Tailscale gives your
phone a private, encrypted path to it from anywhere, with no public URL.

1. **Login for the web UI**

   ```bash
   python3 cli.py set-password            # stores a scrypt hash in .env
   ```

2. **Keep the web UI running** (starts at login, restarts if it stops):

   ```bash
   bash tools/setup_schedule.sh install-web
   ```

3. **Tailscale** — install it on the Mac (tailscale.com/download) and on your
   phone (App Store / Play Store), sign in with the same account on both.

4. **HTTPS on your tailnet** — on the Mac:

   ```bash
   tailscale serve --bg 8050
   ```

   It prints an address like `https://your-mac.tailnet-name.ts.net`. Open that
   on the phone. Because Tailscale now terminates HTTPS in front of the app,
   add these to `.env` and restart the web job (`bash tools/setup_schedule.sh
   install-web` again):

   ```bash
   QUANT_BEHIND_PROXY=1     # trust Tailscale's headers, mark cookies Secure
   QUANT_HOST=127.0.0.1     # Tailscale connects locally; nothing on the Wi-Fi
   ```

5. **Keep the Mac awake** while plugged in: System Settings → Battery →
   Options → *Prevent automatic sleeping on power adapter when the display is
   off*. A sleeping Mac answers nobody, and misses the 7:00 email too.

Only devices signed into your Tailscale account can even see the address;
the app login is a second lock behind it.

## Option B — your Mac + Cloudflare Tunnel (public URL, email login)

Use this if you want a normal web address (on a domain you own, managed by
Cloudflare) that works on any phone without installing anything.

1. `python3 cli.py set-password`, then `bash tools/setup_schedule.sh install-web`.
2. `brew install cloudflared`, then:

   ```bash
   cloudflared tunnel login
   cloudflared tunnel create qrt
   cloudflared tunnel route dns qrt quant.your-domain.com
   cloudflared tunnel run --url http://localhost:8050 qrt
   ```

   (`sudo cloudflared service install` keeps it running.)
3. In the Cloudflare Zero Trust dashboard → Access → Applications, add
   `quant.your-domain.com` with a policy *Emails: your address*. Cloudflare
   then asks for a one-time code sent to your email before the app's own
   login page even loads.
4. Add `QUANT_BEHIND_PROXY=1` and `QUANT_HOST=127.0.0.1` to `.env` and
   restart the web job.

## Option C — an always-on server (no Mac required)

If you would rather not keep a Mac awake, run everything on a small Linux
server (a $4–6/month VPS from Hetzner, DigitalOcean or similar). The included
Docker setup runs the web UI and the 7:00 / 18:30 routine (`cli.py
scheduler`) side by side on one database.

```bash
git clone <your repo> trading && cd trading
cp .env.example .env              # SEC_USER_AGENT, SMTP_*, QUANT_EMAIL, keys…
docker compose build
docker compose run --rm web python cli.py set-password --print
#   → paste the two printed lines into .env
docker compose up -d
```

The port is published on the server's `127.0.0.1` only. Install Tailscale on
the server and run `tailscale serve --bg 8050` there (Option A, steps 3–4), or
point a Cloudflare Tunnel at it (Option B). Add `QUANT_BEHIND_PROXY=1` to
`.env` and `docker compose up -d` again.

**Bringing your data along** instead of re-downloading it:

```bash
python3 cli.py db backup                         # on the Mac
scp db/backups/quant-*.sqlite server:trading/    # newest file
docker compose run --rm -v "$PWD:/import" web \
    sh -c 'cp /import/quant-*.sqlite /app/db/quant.sqlite'
docker compose restart
```

Then remove the Mac's launchd jobs (`bash tools/setup_schedule.sh uninstall`)
so two machines are not both sending the morning email.

The scheduler's times come from `QUANT_TZ` (default `America/New_York`),
`QUANT_MORNING_TIME` (07:00, daily) and `QUANT_EVENING_TIME` (18:30,
weekdays). A server restarted up to three hours after a scheduled time still
runs it; the email itself is never sent twice in a day.

---

## Security notes

- The password is stored as a scrypt hash; changing it (`set-password`
  again) signs every session out.
- Five failed logins from one address lock it out for 15 minutes.
- Sessions are HttpOnly, SameSite=Lax cookies; with `QUANT_BEHIND_PROXY=1`
  they are also Secure (HTTPS only).
- The Control tab can only run its listed commands; nothing typed in a
  browser becomes part of a command line.
- API keys, the SMTP password and your data stay on the machine running the
  app; the phone only ever receives rendered pages.
