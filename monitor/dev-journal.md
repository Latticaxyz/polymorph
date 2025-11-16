# 📓 Monitoring Dev Journal

This journal documents how to deploy, maintain, and understand the monitoring stack for Polymorph scraping + testing.

## Host Roles

| Host                       | Purpose                                       | User      | Notes                           |
| -------------------------- | --------------------------------------------- | --------- | ------------------------------- |
| **Proxmox VM**             | Central monitoring hub (Prometheus + Grafana) | `ubuntu`  | Runs Docker services only       |
| **Ryzen 9 bare-metal box** | Scraping + testing + exporters                | `william` | Runs systemd exporters & runner |

---

## Directory Structure (Repo)

```
monitor/
  docker-compose.yml
  prometheus/prometheus.yml
  exporters/
    node_exporter/
      node_exporter.service
    scraper/
      exporter.py
      exporter.service
      run_scrape.sh
      scraper.service
    tests/
      exporter.py
      tests_exporter.service
```

---

## Deployment Instructions

### 1️⃣ Deploy Monitoring VM (Proxmox → `ubuntu`)

1. Clone repo or sync `monitor/` directory to VM
2. Install **Docker + docker-compose**
3. Run stack:

```bash
cd ~/polymorph/monitor
docker compose up -d
```

4. Access over Tailscale:
   `http://<monitor-vm-ip>:3000` → Grafana
   `http://<monitor-vm-ip>:9090` → Prometheus

---

### 2️⃣ Deploy Exporters + Runner on Ryzen (`william`)

#### Create directories

```bash
sudo mkdir -p /opt/polymorph_exporter
sudo mkdir -p /opt/polymorph_tests_exporter
sudo mkdir -p /data/polymarket
sudo chown -R william:william /opt /data/polymarket
```

#### Copy files from repo

| Repo file                                 | Destination                                 |
| ----------------------------------------- | ------------------------------------------- |
| `monitor/exporters/scraper/exporter.py`   | `/opt/polymorph_exporter/exporter.py`       |
| `monitor/exporters/scraper/run_scrape.sh` | `/usr/local/bin/run_scrape.sh`              |
| `monitor/exporters/tests/exporter.py`     | `/opt/polymorph_tests_exporter/exporter.py` |
| All `.service` files under exporters      | `/etc/systemd/system/`                      |

Ensure permissions:

```bash
chmod +x /usr/local/bin/run_scrape.sh
```

---

### 3️⃣ Systemd Setup (Ryzen)

Enable services in order:

```bash
sudo systemctl daemon-reload
sudo systemctl enable node_exporter.service
sudo systemctl enable exporter.service
sudo systemctl enable tests_exporter.service
sudo systemctl enable scraper.service
```

Start:

```bash
sudo systemctl start node_exporter.service
sudo systemctl start exporter.service
sudo systemctl start tests_exporter.service
sudo systemctl start scraper.service
```

---

## Expected Valid Endpoints

| Host  | URL                              | Purpose         |
| ----- | -------------------------------- | --------------- |
| Ryzen | `http://<ryzen-ip>:9100/metrics` | System metrics  |
| Ryzen | `http://<ryzen-ip>:9400/metrics` | Scraper metrics |
| Ryzen | `http://<ryzen-ip>:9401/metrics` | Tests metrics   |
| VM    | `http://<vm-ip>:9090`            | Prometheus      |
| VM    | `http://<vm-ip>:3000`            | Grafana         |

---

## File & Metric Expectations

### `scraper_state.json`

Must contain:

```json
{
  "run_start": <unix timestamp>,
  "run_end": <unix timestamp>,
  "status": <int>,
  "runs_total": <int>,
  "last_prices_ts": <unix timestamp>,
  "last_trades_ts": <unix timestamp>
}
```

### `tests_state.json`

Must contain:

```json
{
  "last_run_start": <unix timestamp>,
  "last_run_end": <unix timestamp>,
  "tests_total": <int>,
  "tests_failed": <int>,
  "tests_skipped": <int>
}
```

---

## Verification Checklist

- [ ] `docker compose up -d` works on VM
- [ ] Grafana reachable + creds updated
- [ ] Prometheus shows **3 green targets**
- [ ] `systemctl status` on Ryzen shows `active (running)` for all 4 services
- [ ] `scraper_state.json` updates after each run
- [ ] `tests_state.json` updates after each test run
- [ ] Grafana dashboards show non-zero metrics

---

## Notes for Future Devs

- **Repo versions are source of truth** — no editing configs manually on machines.
- Any service or config change must be committed, then redeployed.
- If new exporters are added, update:
  - `monitor/prometheus/prometheus.yml`
  - This journal
  - Grafana dashboards
