# Using the terminal from your phone

**Short version:** your Mac is the server. Install Tailscale on the Mac and
the phone, set a password, and run one command. It's free, takes about ten
minutes, and nothing is ever open to your Wi-Fi or the internet.

## How it is protected

The web UI holds your positions and can send email from your account, so:

- **It listens on this computer only (`127.0.0.1`)**, with or without a
  password. A laptop on the same Wi-Fi gets *connection refused*. It never
  even sees a login page.
- **The phone comes in through Tailscale**, which runs on the Mac and
  forwards to the app locally. Only devices signed into *your* Tailscale
  account can reach the address, and the connection is encrypted end to end.
- **The app's own login is the second lock** (scrypt-hashed password,
  30-day sessions, 5 wrong tries lock that device out for 15 minutes).
- **No password → the tunnel gets nothing.** If Tailscale is started before
  a password is set, the phone sees "set a password first" and no data.
  Only a browser on the Mac itself is served.

Never forward port 8050 on your router. You don't need to.

---

## Set it up (Mac + Tailscale, free)

1. **Set the login** (on the Mac, in the project folder):

   ```bash
   python3 cli.py set-password
   ```

2. **Keep the web UI running** (it starts at login and restarts if it stops):

   ```bash
   bash tools/setup_schedule.sh install-web
   ```

3. **Install Tailscale** on the Mac (tailscale.com/download) and on the
   phone (App Store / Play Store). Sign in with the same account on both.
   The Personal plan is free.

4. **Give the phone an address** (on the Mac):

   ```bash
   tailscale serve --bg 8050
   ```

   The first time, it prints a link to enable HTTPS for your tailnet. Open
   it, click enable, and run the command again. It then prints an address
   like `https://your-mac.tail1234.ts.net`. `--bg` keeps it on across
   reboots. If the `tailscale` command is not found, use
   `/Applications/Tailscale.app/Contents/MacOS/Tailscale serve --bg 8050`.

5. **On the phone**, with Tailscale switched on, open that address and sign
   in. Then **Share → Add to Home Screen**: it opens full-screen like an app
   and stays signed in for 30 days.

6. **Keep the Mac awake while plugged in**: System Settings → Battery →
   Options → *Prevent automatic sleeping on power adapter when the display is
   off*. A sleeping Mac answers nobody, and misses the 7:00 email too.

That's all. No `.env` edits, no router settings, no certificates to manage.
To stop sharing: `tailscale serve --https=443 off`.

The **🎛️ Control** tab is the phone's home screen: market status, last email,
data freshness and whether a scheduled run is in progress, plus buttons to
send today's email, run a full refresh, run AI reviews or back up the
database. Each button asks for confirmation and shows a live log. A button
pressed while the 7:00 job is running waits for it, then does its own check,
so the email is never sent twice.

---

## Do I need a cloud server or a cloud database?

Not for this. The Mac *is* the server, and the setup above is the simplest
free option. The alternatives, measured against this app:

| | Cost | Catch |
|---|---|---|
| **Mac + Tailscale** (above) | free | the Mac must be on and awake |
| Small VPS (Hetzner, DigitalOcean…) | ~$5–8/month | pick **≥ 4 GB RAM**: the daily screens peak at ~2.5 GB on the S&P 500 |
| Oracle Cloud *Always Free* ARM VM | free | enough RAM, but new instances are often "out of capacity" and the free allowance was cut in 2026 |
| Free 1 GB VMs (Google e2-micro, Oracle AMD micro) | free | too little memory for the screens |
| Free web hosts (Render, Hugging Face…) + hosted DB | free | they sleep when idle, so no 7:00 email; the database is ~0.8 GB, over most free DB tiers; and every screen would pull ~3M rows over the internet |

So: Mac + Tailscale if the Mac stays home and plugged in. If you'd rather not
depend on the Mac, a small paid VPS with the Docker setup below.

---

## Alternative: Cloudflare Tunnel (a normal URL, no app on the phone)

For a web address on a domain you own (managed by Cloudflare) that opens on
any phone without installing Tailscale:

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
   login page even loads. Don't skip this step: unlike Tailscale, the
   address is public.

## Alternative: an always-on server (no Mac required)

On a Linux VPS with at least 4 GB of RAM, the included Docker setup runs the
web UI and the 7:00 / 18:30 routine (`cli.py scheduler`) side by side on one
database:

```bash
git clone <your repo> trading && cd trading
cp .env.example .env              # SEC_USER_AGENT, SMTP_*, QUANT_EMAIL, keys…
docker compose build
docker compose run --rm web python cli.py set-password --print
#   → paste the two printed lines into .env (keep the single quotes)
docker compose up -d
```

The port is published on the server's `127.0.0.1` only. Install Tailscale on
the server and run `tailscale serve --bg 8050` there (steps 3–5 above), or
point a Cloudflare Tunnel at it. Times follow `QUANT_TZ` (default
`America/New_York`), including what "today" means for the email.

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

The scheduler's times come from `QUANT_TZ`, `QUANT_MORNING_TIME` (07:00,
daily) and `QUANT_EVENING_TIME` (18:30, weekdays). A server restarted up to
three hours after a scheduled time still runs it. The email itself is never
sent twice in a day.

---

## Security notes

- The password is stored as a scrypt hash. Changing it (`set-password`
  again) signs every session out, and an open page goes back to the login.
- Five failed logins from one device lock that device out for 15 minutes.
  Other devices, including your phone, are unaffected.
- Sessions are HttpOnly, SameSite=Lax cookies, Secure whenever the page was
  loaded over HTTPS (always, through Tailscale or Cloudflare).
- Forwarding headers (the phone's address, "this was HTTPS") are believed
  only from a proxy on the same computer. In Docker, where the host's proxy
  is the only way in, `QUANT_BEHIND_PROXY=1` is set for you.
- Without a password, only a browser on the computer itself is served. Requests
  through a tunnel, from the network, or from a web page trying DNS
  rebinding are refused.
- The Control tab can only run its listed commands. Nothing typed in a
  browser becomes part of a command line.
- API keys, the SMTP password and your data stay on the machine running the
  app. The phone only ever receives rendered pages.
- Advanced: `QUANT_HOST=0.0.0.0` makes the app listen on your Wi-Fi directly
  (it refuses without a password). Not needed with any of the options above.
