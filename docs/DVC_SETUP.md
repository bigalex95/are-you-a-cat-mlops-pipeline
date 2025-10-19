# 📦 DVC Multi-Remote Setup Guide

## 🎯 Introduction

This project uses **DVC (Data Version Control)** with a **multi-remote storage setup** combining **Backblaze B2** and **DagsHub**. This configuration ensures:

- ✅ **Redundancy**: Data is stored in multiple locations
- ✅ **Bandwidth Management**: Avoid hitting free tier limits
- ✅ **High Availability**: Automatic fallback if one remote is unavailable
- ✅ **Cost Efficiency**: Leverage free tiers from both providers

### 🗄️ Why Multiple Remotes?

Using multiple DVC remotes provides several advantages:

1. **Backup & Redundancy**: If one service is down, you can still access your data
2. **Bandwidth Optimization**: Switch between remotes when hitting daily download limits
3. **Performance**: Choose the fastest remote for your location
4. **Disaster Recovery**: Multiple copies protect against data loss
5. **Flexibility**: Different team members can use different remotes based on their needs

---

## 📊 Free Tier Comparison

| Feature | Backblaze B2 | DagsHub |
|---------|--------------|---------|
| **Storage** | 10 GB | 10 GB |
| **Download Limit** | 1 GB/day ⚠️ | Unlimited ✅ |
| **Upload Speed** | Very Fast 🚀 | Moderate |
| **Download Speed** | Very Fast 🚀 | Moderate |
| **Best For** | Daily development | Heavy downloads |

### ⚠️ Backblaze Free Tier Limits

- **Storage**: 10 GB (sufficient for most ML projects)
- **Daily Download**: 1 GB/day limit
- **Monthly Download**: ~3x storage (~30 GB/month)
- If you exceed the daily limit, you'll need to wait 24 hours or switch to DagsHub

### ✅ DagsHub Free Tier Limits

- **Storage**: 10 GB
- **Downloads**: Unlimited (no daily caps!)
- **Perfect fallback** when Backblaze limits are reached

---

## 🔧 Prerequisites

Before setting up DVC with multiple remotes, ensure you have:

- ✅ **Python 3.10+** installed
- ✅ **Git** configured and initialized
- ✅ **DVC** installed: `pip install dvc dvc[s3]`
- ✅ **Existing Backblaze B2 setup** (bucket created, credentials ready)
- ✅ **GitHub account** (to sign up for DagsHub)

---

## 🚀 Step-by-Step Setup Instructions

### Step 1: Rename Existing Remote to `backblaze`

If you already have a DVC remote named `myremote`, rename it for clarity:

```bash
# Check current remotes
dvc remote list

# Rename myremote to backblaze
dvc remote rename myremote backblaze

# Verify the change
dvc remote list
```

**Expected Output:**
```
backblaze       b2://your-bucket-name
```

### Step 2: Sign Up for DagsHub

1. Go to [https://dagshub.com/](https://dagshub.com/)
2. Click "Sign Up" (free account)
3. Sign up with your GitHub account (recommended for quick setup)
4. Verify your email address

### Step 3: Create/Connect Repository on DagsHub

**Option A: Connect Existing GitHub Repository**

1. Click "New Repository" on DagsHub
2. Select "Connect an existing repository"
3. Choose GitHub as the source
4. Select your repository: `bigalex95/are-you-a-cat-mlops-pipeline`
5. Click "Create Repository"

**Option B: Create New Repository**

1. Click "New Repository"
2. Name: `are-you-a-cat-mlops-pipeline`
3. Description: Optional
4. Visibility: Public or Private
5. Click "Create Repository"

### Step 4: Generate DagsHub Access Token

1. Go to [https://dagshub.com/user/settings/tokens](https://dagshub.com/user/settings/tokens)
2. Click "New Token"
3. Name: `dvc-access` (or any descriptive name)
4. Scopes: Select "read:dvc" and "write:dvc"
5. Click "Generate Token"
6. **⚠️ IMPORTANT**: Copy the token immediately (it won't be shown again!)

### Step 5: Add DagsHub as Second Remote

```bash
# Add DagsHub remote (not as default)
dvc remote add dagshub https://dagshub.com/bigalex95/are-you-a-cat-mlops-pipeline.dvc

# Verify both remotes exist
dvc remote list
```

**Expected Output:**
```
backblaze       b2://your-bucket-name
dagshub         https://dagshub.com/bigalex95/are-you-a-cat-mlops-pipeline.dvc
```

### Step 6: Configure DagsHub Authentication

**⚠️ Store credentials locally (not in Git):**

```bash
# Configure authentication method
dvc remote modify dagshub --local auth basic

# Set your DagsHub username
dvc remote modify dagshub --local user bigalex95

# Set your DagsHub token (paste the token you generated)
dvc remote modify dagshub --local password YOUR_DAGSHUB_TOKEN
```

**💡 Why `--local`?**
- Credentials are stored in `.dvc/config.local` (not tracked by Git)
- Keeps secrets out of version control
- Each team member configures their own credentials

### Step 7: Set Backblaze as Default Remote

```bash
# Set backblaze as the default remote
dvc remote default backblaze

# Verify configuration
dvc remote list
dvc remote default
```

**Expected Output:**
```
backblaze       b2://your-bucket-name
dagshub         https://dagshub.com/bigalex95/are-you-a-cat-mlops-pipeline.dvc

Default remote: backblaze
```

### Step 8: Initial Data Push to DagsHub

Push your existing data to DagsHub:

```bash
# Push to DagsHub explicitly
dvc push -r dagshub

# Or use the sync script (pushes to all remotes)
./scripts/dvc_sync_remotes.sh
```

### Step 9: Verify Setup

Test that both remotes work:

```bash
# Check DVC status
dvc status

# Test pull from Backblaze
dvc pull -r backblaze --dry-run

# Test pull from DagsHub
dvc pull -r dagshub --dry-run
```

---

## 🔄 Daily Usage Workflows

### Scenario 1: Normal Development (Backblaze Available)

**Morning: Start Work**
```bash
# Smart pull (tries Backblaze, falls back to DagsHub if needed)
./scripts/dvc_pull_smart.sh

# Or explicitly from Backblaze
dvc pull -r backblaze
```

**During Work**
```bash
# Make changes to data/models
dvc add data/new_dataset.csv

# Track changes in Git
git add data/new_dataset.csv.dvc
git commit -m "Add new dataset"
```

**End of Day: Push Changes**
```bash
# Push to all remotes (recommended)
./scripts/dvc_push_all.sh

# Or push individually
dvc push -r backblaze
dvc push -r dagshub
```

### Scenario 2: Backblaze Limit Reached (1GB/day)

**When you see:**
```
ERROR: failed to pull data from the cloud - ...
ERROR: ... bandwidth limit exceeded ...
```

**Switch to DagsHub:**
```bash
# Pull from DagsHub instead (unlimited downloads)
dvc pull -r dagshub

# Continue working normally
# ...

# Push back to DagsHub
dvc push -r dagshub
```

**Next day:**
```bash
# Sync data back to Backblaze when limit resets
dvc push -r backblaze
```

### Scenario 3: Syncing Both Remotes

**Keep both remotes in sync:**
```bash
# Use the sync script
./scripts/dvc_sync_remotes.sh

# Or manually
dvc push -r backblaze
dvc push -r dagshub
```

---

## 🛠️ Helper Scripts Usage

### `scripts/dvc_pull_smart.sh`

**Purpose**: Intelligent data pulling with automatic fallback

**Usage:**
```bash
./scripts/dvc_pull_smart.sh
```

**How it works:**
1. First tries to pull from Backblaze (fast, but limited)
2. If Backblaze fails (e.g., bandwidth limit), automatically falls back to DagsHub
3. Provides clear status messages at each step

**When to use:**
- ✅ Start of workday
- ✅ After pulling new code with updated DVC files
- ✅ When unsure which remote to use

### `scripts/dvc_push_all.sh`

**Purpose**: Push data to all configured remotes

**Usage:**
```bash
./scripts/dvc_push_all.sh
```

**How it works:**
1. Pushes to Backblaze
2. Pushes to DagsHub
3. Reports success/failure for each remote
4. Returns non-zero exit code if any push fails

**When to use:**
- ✅ After adding new data/models
- ✅ End of workday to ensure redundancy
- ✅ Before sharing work with team

### `scripts/dvc_sync_remotes.sh`

**Purpose**: Ensure both remotes have identical data

**Usage:**
```bash
./scripts/dvc_sync_remotes.sh
```

**How it works:**
1. Pushes all tracked data to Backblaze
2. Pushes all tracked data to DagsHub
3. Stops on first error

**When to use:**
- ✅ After recovering from Backblaze bandwidth limit
- ✅ Weekly maintenance to ensure redundancy
- ✅ Before team collaborations

---

## 🐛 Troubleshooting

### Problem: "Authentication failed" for DagsHub

**Solution:**
```bash
# Reconfigure DagsHub credentials
dvc remote modify dagshub --local user YOUR_USERNAME
dvc remote modify dagshub --local password YOUR_NEW_TOKEN
```

### Problem: "Backblaze bandwidth limit exceeded"

**Solution:**
```bash
# Use DagsHub instead
dvc pull -r dagshub

# Or use smart pull (automatic fallback)
./scripts/dvc_pull_smart.sh
```

### Problem: "Remote not found"

**Solution:**
```bash
# List configured remotes
dvc remote list

# Add missing remote
dvc remote add dagshub https://dagshub.com/bigalex95/are-you-a-cat-mlops-pipeline.dvc
```

### Problem: "Push fails silently"

**Solution:**
```bash
# Enable verbose output
dvc push -r backblaze -v
dvc push -r dagshub -v

# Check DVC status
dvc status
```

### Problem: Scripts don't run on Windows Git Bash

**Solution:**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run with bash explicitly
bash scripts/dvc_pull_smart.sh
```

### Problem: "File not found" in DagsHub

**Solution:**
```bash
# Ensure data was pushed to DagsHub
dvc push -r dagshub

# Check DagsHub web UI: https://dagshub.com/bigalex95/are-you-a-cat-mlops-pipeline/dvc
```

---

## 💡 Best Practices

### 1. **Always Use Smart Pull**

```bash
# Good: Automatic fallback
./scripts/dvc_pull_smart.sh

# Okay: Manual remote selection
dvc pull -r backblaze
```

### 2. **Push to Both Remotes Regularly**

```bash
# At end of workday
./scripts/dvc_push_all.sh
```

### 3. **Monitor Backblaze Usage**

- Track your daily downloads (check Backblaze dashboard)
- Switch to DagsHub proactively if approaching 1GB
- Reset occurs at midnight Pacific Time

### 4. **Use DagsHub for Large Downloads**

```bash
# Downloading full dataset for first time
dvc pull -r dagshub

# Then use Backblaze for incremental updates
dvc pull -r backblaze
```

### 5. **Keep Credentials Secure**

```bash
# Never commit .dvc/config.local
echo ".dvc/config.local" >> .gitignore

# Use environment variables in CI/CD
export DAGSHUB_TOKEN=your_token
```

### 6. **Verify Before Deleting Local Data**

```bash
# Ensure data is in both remotes
./scripts/dvc_sync_remotes.sh

# Verify with dry-run
dvc pull -r backblaze --dry-run
dvc pull -r dagshub --dry-run
```

### 7. **Document Remote Selection**

In team settings, document which remote to use:

```bash
# In your README or commit messages
# Using Backblaze for speed
dvc pull -r backblaze

# Switched to DagsHub due to bandwidth
dvc pull -r dagshub
```

---

## 📚 Additional Resources

- **DVC Documentation**: [https://dvc.org/doc](https://dvc.org/doc)
- **DVC Multi-Remote Setup**: [https://dvc.org/doc/user-guide/data-management/remote-storage#multiple-remotes](https://dvc.org/doc/user-guide/data-management/remote-storage#multiple-remotes)
- **Backblaze B2 Docs**: [https://www.backblaze.com/b2/docs/](https://www.backblaze.com/b2/docs/)
- **DagsHub Documentation**: [https://dagshub.com/docs](https://dagshub.com/docs)
- **DVC with DagsHub**: [https://dagshub.com/docs/integration_guide/dvc/](https://dagshub.com/docs/integration_guide/dvc/)

---

## 🤝 Need Help?

If you encounter issues not covered in this guide:

1. Check DVC status: `dvc status`
2. Enable verbose mode: `dvc pull -v` or `dvc push -v`
3. Check DagsHub web UI for uploaded files
4. Review Backblaze dashboard for bandwidth usage
5. Open an issue on GitHub

---

**Happy Data Versioning! 🚀**