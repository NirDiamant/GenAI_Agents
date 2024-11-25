import { ocr } from 'llama-ocr';
import fs from 'fs/promises';

// Fetch the API key from the environment variable
const apiKey = process.env.LLAMA_OCR_API_KEY;

async function getMarkdownAndSave() {
  try {
    const markdown = await ocr({
      filePath: "g1.png",
      apiKey: apiKey
    });

    // Save the extracted markdown to a file
    const filePath = "../extracted/grocery_receipt.md";
    await fs.writeFile(filePath, markdown, "utf8");

    console.log(`Markdown saved to ${filePath}`);
  } catch (error) {
    console.error("Error saving markdown:", error);
  }
}

// Call the function
getMarkdownAndSave();