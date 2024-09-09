# Contributing to GenAI Agents Playground

Welcome to the world's most comprehensive repository of Generative AI Agent tutorials and implementations! 🌟 We're thrilled you're interested in contributing to this dynamic knowledge base. Your expertise and creativity can help us push the boundaries of GenAI agent technology.

## Join Our Community

We have a vibrant Discord community where contributors can discuss ideas, ask questions, and collaborate on GenAI agent techniques. Join us at:

[GenAI Agents Discord Server](https://discord.gg/cA6Aa4uyDX)

Don't hesitate to introduce yourself and share your thoughts!

## Ways to Contribute

We welcome contributions of all kinds! Here are some ways you can help:

1. **Add New GenAI Agents:** Create new notebooks showcasing novel agent implementations.
2. **Improve Existing Notebooks:** Enhance, update, or expand our current tutorials.
3. **Fix Bugs:** Help us squash bugs in existing code or explanations.
4. **Enhance Documentation:** Improve clarity, add examples, or fix typos in our docs.
5. **Share Creative Ideas:** Have an innovative idea for a new agent? We're all ears!
6. **Engage in Discussions:** Participate in our Discord community to help shape the future of GenAI agents.

Remember, no contribution is too small. Every improvement helps make this repository an even better resource for the community.

## Reporting Issues

Found a problem or have a suggestion? Please create an issue on GitHub, providing as much detail as possible. You can also discuss issues in our Discord community.

## Contributing Code or Content

1. **Fork and Branch:** Fork the repository and create your branch from `main`.
2. **Make Your Changes:** Implement your contribution, following our best practices.
3. **Test:** Ensure your changes work as expected.
4. **Follow the Style:** Adhere to the coding and documentation conventions used throughout the project.
5. **Commit:** Make your git commits informative and concise.
6. **Stay Updated:** The main branch is frequently updated. Before opening a pull request, make sure your code is up-to-date with the current main branch and has no conflicts.
7. **Push and Pull Request:** Push to your fork and submit a pull request.
8. **Discuss:** Use the Discord community to discuss your contribution if you need feedback or have questions.

## Adding a New GenAI Agent

When adding a new GenAI agent to the repository, please follow these additional steps:

1. Create your notebook in the `all_agents_tutorials` folder.
2. Update the README.md file:
   - Add your new agent to the list of implementations in the README.
   - Place it in the appropriate category based on complexity and type.
   - Use the following format for the link:
     ```
     ### [Number]. [Your Agent Name 🏷️](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/your_file_name.ipynb)
     ```
   - Replace `[Number]` with the appropriate number, `[Your Agent Name]` with your agent's name, and `your_file_name.ipynb` with the actual name of your notebook file.
   - Choose an appropriate emoji that represents your agent.
   - After inserting your new agent, make sure to update the numbers of all subsequent agents to maintain the correct order.

For example:
```
### 🌱 Beginner-Friendly Agents

1. [Simple Conversational Agent 💬](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_conversational_agent.ipynb)
2. [Your New Agent 🆕](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/your_new_agent.ipynb)
3. [Next Agent 🔜](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/next_agent.ipynb)
```

Remember to increment the numbers of all agents that come after your newly inserted implementation.

## Notebook Structure

For new notebooks or significant additions to existing ones, please follow this structure:

1. **Title and Overview:** Clear title and brief overview of the agent.

2. **Detailed Explanation:** Cover motivation, key components, agent architecture, and benefits.

3. **Visual Representation:** Include a diagram to visualize the agent's architecture. We recommend using Mermaid syntax for creating these diagrams. Here's how to do it:

• Create a graph using Mermaid's graph TD (top-down) syntax<br>
• You can use Claude or other AI assistants to help you design the graph if needed<br>
• Paste your Mermaid code into [Mermaid Live Editor](https://mermaid.live/)<br>
• In the "Actions" tab of Mermaid Live Editor, download the SVG file of your diagram<br>
• Store the SVG file in the [images folder](https://github.com/NirDiamant/GenAI_Agents/tree/main/images) of the repository<br>
• Use an appropriate, descriptive name for the file<br>
• In your notebook, display the image using Markdown syntax:<br>
  ```markdown
  ![Your Agent Name](../images/your-agent-name.svg)
  ```

This process ensures consistency in our visual representations and makes it easy for others to understand and potentially modify the diagrams in the future.

4. **Implementation:** Step-by-step Python implementation with clear comments and explanations.

5. **Usage Example:** Demonstrate the agent with a practical example.

6. **Comparison:** Compare with simpler agents, both qualitatively and quantitatively if possible.

7. **Additional Considerations:** Discuss limitations, potential improvements, or specific use cases.

8. **References:** Include relevant citations or resources if you have.

## Notebook Best Practices

To ensure consistency and readability across all notebooks:

1. **Code Cell Descriptions:** Each code cell should be preceded by a markdown cell with a clear, concise title describing the cell's content or purpose.

2. **Clear Unnecessary Outputs:** Before committing your notebook, clear all unnecessary cell outputs. This helps reduce file size and avoids confusion from outdated results.

3. **Consistent Formatting:** Maintain consistent formatting throughout the notebook, including regular use of markdown headers, code comments, and proper indentation.

## Code Quality and Readability

To ensure the highest quality and readability of our code:

1. **Write Clean Code:** Follow best practices for clean, readable code.
2. **Use Comments:** Add clear and concise comments to explain complex logic.
3. **Format Your Code:** Use consistent formatting throughout your contribution.
4. **Language Model Review:** After completing your code, consider passing it through a language model for additional formatting and readability improvements. This extra step can help make your code even more accessible and maintainable.

## Documentation

Clear documentation is crucial. Whether you're improving existing docs or adding new ones, follow the same process: fork, change, test, and submit a pull request.

## Final Notes

We're grateful for all our contributors and excited to see how you'll help expand the world's most comprehensive GenAI agent resource. Don't hesitate to ask questions in our Discord community if you're unsure about anything.

Let's harness our collective knowledge and creativity to push the boundaries of GenAI agent technology together!

Happy contributing! 🚀