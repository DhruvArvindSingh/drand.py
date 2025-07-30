# Claude AI Code Generation Workflow Setup

This repository includes a GitHub Actions workflow that uses Claude AI via Amazon Bedrock to automatically generate code based on issue descriptions or comments.

## Overview

The workflow can be triggered in three ways:
1. **Issues**: Create an issue with the `claude-generate` label
2. **Comments**: Comment `/claude generate <description>` on any issue
3. **Manual**: Use the "Actions" tab to manually trigger the workflow

## Setup Requirements

### 1. AWS Bedrock Access

You need AWS credentials with access to Amazon Bedrock and Claude models:

```bash
# Required AWS permissions:
- bedrock:InvokeModel
- bedrock:ListFoundationModels
```

### 2. GitHub Secrets

Add these secrets to your repository (Settings → Secrets and variables → Actions):

```
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1  # or your preferred region with Bedrock access
AWS_BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0  # optional, defaults to Claude 3 Sonnet
```

### 3. Repository Permissions

Ensure the workflow has the necessary permissions:
- Contents: write (to create branches)
- Pull requests: write (to create PRs)
- Issues: write (to comment on issues)

## Usage Examples

### Method 1: Issue with Label

1. Create a new issue
2. Add the `claude-generate` label
3. Describe your feature request in the issue body:

```
Add a new function to get multiple random values in parallel

I need a function that can fetch random values for multiple rounds simultaneously 
to improve performance when getting historical randomness data.

Requirements:
- Should accept a list of round numbers
- Return results in the same order as requested
- Handle errors gracefully for invalid rounds
- Use async/await for concurrent requests
```

### Method 2: Comment Command

Comment on any existing issue:

```
/claude generate Add caching functionality to reduce API calls to drand servers
```

### Method 3: Manual Workflow Dispatch

1. Go to the "Actions" tab in your repository
2. Select "Claude AI Code Generation"
3. Click "Run workflow"
4. Fill in the task description and optional target branch

## What the Workflow Does

1. **Analyzes** your Python codebase structure and existing patterns
2. **Generates** code using Claude AI with context about your drand.py project
3. **Creates** a new branch with the generated code
4. **Opens** a pull request with:
   - Generated code changes
   - Implementation notes
   - Testing checklist
   - Review guidelines

## Generated Pull Request Features

Each generated PR includes:

- **Code Changes**: New files or modifications to existing files
- **Documentation**: Clear explanation of what was implemented
- **Testing Guide**: Instructions for testing the new functionality
- **Review Checklist**: Items to verify before merging
- **Compatibility Notes**: Ensuring integration with existing codebase

## Example Generated Code

The workflow understands your drand.py project context and will generate code that:

- ✅ Uses async/await patterns (compatible with aiohttp)
- ✅ Follows PEP 8 style guidelines
- ✅ Includes proper error handling
- ✅ Adds type hints where beneficial
- ✅ Includes docstrings for new functions
- ✅ Maintains cryptographic security best practices
- ✅ Includes unit tests when appropriate

## Workflow Customization

### Changing the Claude Model

Update the `AWS_BEDROCK_MODEL_ID` secret to use different Claude models:

```
# Claude 3 Haiku (faster, less expensive)
anthropic.claude-3-haiku-20240307-v1:0

# Claude 3 Sonnet (balanced)
anthropic.claude-3-sonnet-20240229-v1:0

# Claude 3 Opus (most capable)
anthropic.claude-3-opus-20240229-v1:0
```

### Modifying the Prompt

Edit the `generate_code_with_claude` function in the workflow to customize:
- Code generation guidelines
- Project-specific requirements
- Output format preferences

## Troubleshooting

### Common Issues

1. **AWS Credentials**: Ensure your AWS credentials have Bedrock access
2. **Region**: Make sure you're using a region where Bedrock is available
3. **Model Access**: Verify you have access to the Claude model in Bedrock console
4. **Branch Conflicts**: The workflow creates new branches, ensure names don't conflict

### Debugging

Check the workflow logs in the "Actions" tab for detailed error messages:
- AWS authentication issues
- Bedrock API errors
- Git operation failures
- Pull request creation problems

## Security Considerations

- AWS credentials are stored as GitHub secrets (encrypted)
- Generated code should always be reviewed before merging
- The workflow runs in an isolated GitHub Actions environment
- No sensitive data from your repository is sent to Claude (only code structure)

## Best Practices

1. **Be Specific**: Provide detailed task descriptions for better results
2. **Review Carefully**: Always review generated code before merging
3. **Test Thoroughly**: Run the provided test commands
4. **Iterate**: Use the feedback loop to refine requests
5. **Security First**: Pay special attention to cryptographic code changes

## Support

If you encounter issues:
1. Check the workflow logs in GitHub Actions
2. Verify your AWS Bedrock setup
3. Ensure all required secrets are configured
4. Review the generated PR for any error messages

---

*This workflow leverages Claude AI's understanding of Python, cryptography, and distributed systems to generate contextually appropriate code for your drand.py project.*
