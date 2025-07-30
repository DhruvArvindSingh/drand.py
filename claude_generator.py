import boto3
import json
import os
import requests
import subprocess
import tempfile
from botocore.config import Config


def get_bedrock_client():
    config = Config(
        read_timeout=120,
        connect_timeout=10,
        retries={
            'max_attempts': 3,
            'mode': 'standard'
        }
    )
    return boto3.client('bedrock-runtime', region_name=os.environ['AWS_DEFAULT_REGION'], config=config)

def read_file_safely(filepath, max_lines=100):
    """Read file content safely, limiting lines for context"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > max_lines:
                return ''.join(lines[:max_lines]) + f'\n... (truncated, {len(lines)-max_lines} more lines)'
            return ''.join(lines)
    except Exception as e:
        return f"Error reading file: {str(e)}"

def analyze_codebase():
    """Analyze the current Python codebase structure"""
    context = ""
    
    # Read project context
    if os.path.exists('project_context.txt'):
        context += read_file_safely('project_context.txt')
    
    # Read key configuration files
    key_files = ['setup.py', 'pyproject.toml', 'requirements.txt', 'Makefile', 'README.rst', 'README.md']
    for file in key_files:
        if os.path.exists(file):
            context += f"\n\n=== {file} ===\n"
            context += read_file_safely(file, 50)
    
    # Sample some source files to understand patterns
    if os.path.exists('codebase_files.txt'):
        with open('codebase_files.txt', 'r') as f:
            files = [line.strip() for line in f.readlines()[:15]]  # First 15 files
        
        for file in files:
            if os.path.exists(file):
                context += f"\n\n=== {file} ===\n"
                context += read_file_safely(file, 40)
    
    return context

def generate_code_with_claude(task_description, codebase_context):
    client = get_bedrock_client()
    
    prompt = f"""
You are an expert Python developer working on a drand.py project - a Python client for the drand distributed randomness beacon network.

TASK: {task_description}

CURRENT CODEBASE CONTEXT:
{codebase_context}

Please generate the necessary code changes to implement the requested feature. Your response should include:

1. **FILES_TO_CREATE**: List any new files that need to be created with their full paths
2. **FILES_TO_MODIFY**: List any existing files that need to be modified
3. **CODE_CHANGES**: Provide the actual code for new files or specific changes for existing files
4. **INSTRUCTIONS**: Any additional setup or configuration steps needed

Follow these guidelines:
- Use Python 3.6+ compatible syntax
- Follow PEP 8 style guidelines
- Use async/await patterns where appropriate (the project uses aiohttp)
- Include proper error handling and exceptions
- Add appropriate type hints where beneficial
- Include docstrings for new functions and classes
- Use existing utilities and patterns from the codebase
- Ensure compatibility with the existing drand network protocol
- Include unit tests for new functionality when appropriate
- Consider cryptographic security best practices (the project deals with cryptographic verification)

Format your response clearly with sections for each file change.
"""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8000,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    response = client.invoke_model(
        body=json.dumps(body),
        modelId=os.environ.get('AWS_BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0'),
        accept='application/json',
        contentType='application/json'
    )
    
    response_body = json.loads(response.get('body').read())
    return response_body['content'][0]['text']

def create_branch_and_commit(branch_name, generated_content):
    """Create a new branch and commit the generated code"""
    try:
        subprocess.run(['git', 'config', 'user.name', 'Claude Bot'], check=True)
        subprocess.run(['git', 'config', 'user.email', 'claude-bot@example.com'], check=True)

        # Create and checkout new branch
        subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
        
        # Create a summary file with the generated content
        with open('CLAUDE_GENERATED.md', 'w') as f:
            f.write(f"# Claude Generated Code\n\n")
            f.write(f"**Task**: {os.environ['TASK_DESCRIPTION']}\n\n")
            f.write(f"**Generated on**: {subprocess.check_output(['date']).decode().strip()}\n\n")
            f.write("## Generated Content\n\n")
            f.write("```\n")
            f.write(generated_content)
            f.write("\n```\n")
            f.write("\n## Implementation Notes\n\n")
            f.write("- Review the generated code carefully before merging\n")
            f.write("- Run tests to ensure compatibility: `python -m pytest tests/`\n")
            f.write("- Check code style: `flake8 drand/`\n")
            f.write("- Verify cryptographic functions work correctly\n")
        
        # Stage and commit changes
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', f'Add Claude generated code\n\nTask: {os.environ["TASK_DESCRIPTION"]}\n\nGenerated by Claude AI via Amazon Bedrock'], check=True)
        
        # Push branch
        subprocess.run(['git', 'push', '-u', 'origin', branch_name], check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Git operation failed: {e}")
        return False

def create_pull_request(branch_name, task_description, generated_content):
    """Create a pull request with the generated code"""
    github_token = os.environ['GITHUB_TOKEN']
    repo = os.environ['GITHUB_REPOSITORY']
    
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    pr_body = f"""## 🤖 Claude AI Generated Code

**Task Description:** {task_description}

This pull request contains code generated by Claude AI via Amazon Bedrock based on the requested feature for the drand.py project.

## Generated Changes

```
{generated_content[:2000]}{'...' if len(generated_content) > 2000 else ''}
```

## Review Checklist

- [ ] Review the generated code for correctness
- [ ] Run tests: `python -m pytest tests/`
- [ ] Check code style: `flake8 drand/`
- [ ] Verify cryptographic functions work correctly
- [ ] Test with local drand devnet if applicable
- [ ] Check compatibility with existing API
- [ ] Ensure proper error handling
- [ ] Verify async/await patterns are correct

## Testing

To test the changes:

```bash
# Install in development mode
pip install -e .[dev]

# Run tests
python -m pytest tests/

# Run local devnet (if needed)
cd devnet && ./run.sh
```

---
*This PR was created automatically by Claude AI*
"""
    
    pr_data = {
        'title': f'🤖 Claude Generated: {task_description[:50]}{"..." if len(task_description) > 50 else ""}',
        'head': branch_name,
        'base': 'master',  # drand.py uses master as default branch
        'body': pr_body
    }
    
    url = f'https://api.github.com/repos/{repo}/pulls'
    response = requests.post(url, headers=headers, json=pr_data)
    
    if response.status_code == 201:
        pr_url = response.json()['html_url']
        print(f"✅ Pull request created: {pr_url}")
        return pr_url
    else:
        print(f"❌ Failed to create PR: {response.status_code}")
        print(response.text)
        return None

def main():
    task_description = os.environ['TASK_DESCRIPTION']
    branch_name = os.environ['TARGET_BRANCH']
    
    print(f"🚀 Generating code for drand.py task: {task_description}")
    
    # Analyze codebase
    print("📊 Analyzing Python codebase...")
    codebase_context = analyze_codebase()
    
    # Generate code with Claude
    print("🧠 Generating code with Claude...")
    generated_content = generate_code_with_claude(task_description, codebase_context)
    
    # Create branch and commit
    print(f"🌿 Creating branch: {branch_name}")
    if create_branch_and_commit(branch_name, generated_content):
        # Create pull request
        print("📝 Creating pull request...")
        pr_url = create_pull_request(branch_name, task_description, generated_content)
        
        if pr_url:
            print(f"🎉 Code generation completed successfully!")
            print(f"Pull request: {pr_url}")
        else:
            print("⚠️ Code generated but PR creation failed")
    else:
        print("❌ Failed to create branch and commit changes")

if __name__ == "__main__":
    main()
