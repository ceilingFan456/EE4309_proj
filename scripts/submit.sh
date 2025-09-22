#!/usr/bin/env bash
set -euo pipefail

# EE4309 ViTDet Project Submission Script
# This script helps students submit their work with proper git commit

echo "================================================"
echo "    EE4309 ViTDet Project Submission Tool"
echo "================================================"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Error: Git repository not found!"
    echo "This project should contain a .git folder with the initial code."
    echo "Please contact your instructor if the .git folder is missing."
    exit 1
fi

# Check for uncommitted changes
if [ -z "$(git status --porcelain)" ]; then
    echo "⚠️  No changes detected. Your work might already be submitted."
    echo ""
    echo "📝 Previous submissions:"
    git log --oneline -5
    echo ""
    read -p "Do you want to create a new submission anyway? (y/N): " FORCE_SUBMIT
    if [[ ! "$FORCE_SUBMIT" =~ ^[Yy]$ ]]; then
        echo "Submission cancelled."
        exit 0
    fi
fi

# Get student information
echo "📝 Please enter your information:"
echo ""

# Get student name
while true; do
    read -p "Enter your full name: " STUDENT_NAME
    if [ -z "$STUDENT_NAME" ]; then
        echo "❌ Name cannot be empty. Please try again."
    else
        break
    fi
done

# Get student ID
while true; do
    read -p "Enter your student ID (e.g., A0123456X): " STUDENT_ID
    if [ -z "$STUDENT_ID" ]; then
        echo "❌ Student ID cannot be empty. Please try again."
    elif [[ ! "$STUDENT_ID" =~ ^[A-Z][0-9]{7}[A-Z]$ ]]; then
        echo "❌ Invalid student ID format. Should be like A0123456X. Please try again."
    else
        break
    fi
done

# Check if this student has already submitted
EXISTING_SUBMISSION=$(git log --grep="Student ID: ${STUDENT_ID}" --oneline | head -1)
if [ ! -z "$EXISTING_SUBMISSION" ]; then
    echo ""
    echo "⚠️  Warning: Found existing submission for ${STUDENT_ID}:"
    echo "   ${EXISTING_SUBMISSION}"
    echo ""
    read -p "Do you want to create a new submission? (y/N): " CONFIRM_NEW
    if [[ ! "$CONFIRM_NEW" =~ ^[Yy]$ ]]; then
        echo "Submission cancelled."
        exit 0
    fi
fi

# Get optional message
read -p "Enter additional message (optional, press Enter to skip): " ADDITIONAL_MSG

echo ""
echo "================================================"
echo "📦 Preparing submission..."
echo "================================================"

# Configure git user (local to this repository only)
git config user.name "$STUDENT_NAME"
git config user.email "${STUDENT_ID}@u.nus.edu"

# Show changes to be committed
echo ""
echo "📊 Changes to be submitted:"
echo "----------------------------"
git diff --stat
echo "----------------------------"
echo ""

# Show detailed file changes
echo "📝 Modified files:"
git status --short

# Stage all changes
echo ""
echo "📂 Staging all changes..."
git add -A

# Create commit message
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
COMMIT_MSG="[SUBMISSION] ${STUDENT_NAME} (${STUDENT_ID})

Student Name: ${STUDENT_NAME}
Student ID: ${STUDENT_ID}
Submission Time: ${TIMESTAMP}"

if [ ! -z "$ADDITIONAL_MSG" ]; then
    COMMIT_MSG="${COMMIT_MSG}
Additional Notes: ${ADDITIONAL_MSG}"
fi

# Show commit message preview
echo ""
echo "📄 Commit message preview:"
echo "----------------------------"
echo "$COMMIT_MSG"
echo "----------------------------"
echo ""

# Confirm submission
read -p "Confirm submission? (Y/n): " CONFIRM
if [[ "$CONFIRM" =~ ^[Nn]$ ]]; then
    echo "Submission cancelled."
    git reset HEAD
    exit 0
fi

# Create commit
echo ""
echo "💾 Creating commit..."
git commit -m "$COMMIT_MSG" || {
    echo ""
    echo "❌ Commit failed. Please check the error message above."
    exit 1
}

echo ""
echo "================================================"
echo "✅ Submission Successful!"
echo "================================================"
echo ""
echo "📌 Your submission has been recorded:"
echo "  👤 Student Name: $STUDENT_NAME"
echo "  🆔 Student ID: $STUDENT_ID"
echo "  🕐 Timestamp: $TIMESTAMP"
echo ""

# Show how to view changes
echo "================================================"
echo "📢 Verification Commands:"
echo "================================================"
echo "View your changes:        git diff HEAD~1"
echo "View submission history:  git log --oneline"
echo "View detailed history:    git log --stat"
echo ""

echo "================================================"
echo "⚠️  Important Reminders:"
echo "================================================"
echo "1. DO NOT modify the .git folder"
echo "2. DO NOT delete or reset commits"
echo "3. Keep this repository until grading is complete"
echo "4. You can make multiple submissions if needed"
echo ""
echo "🎉 Good luck with your project!"