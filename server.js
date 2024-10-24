const express = require("express");
const multer = require("multer");
const { DocxLoader } = require("@langchain/community/document_loaders/fs/docx");
const { PDFLoader } = require("@langchain/community/document_loaders/fs/pdf");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { AzureOpenAIEmbeddings } = require("@langchain/openai"); // Import Azure OpenAI Embeddings
const fs = require("fs");
const path = require("path");
const mongoose = require("mongoose");
const { ChatOpenAI } = require("@langchain/openai"); // Import ChatOpenAI
const cors = require("cors"); // Import CORS
const pdfParse = require("pdf-parse");

require('dotenv').config();


// Initialize Azure OpenAI embeddings
const embeddings = new AzureOpenAIEmbeddings({
  azureOpenAIApiKey: process.env.AZURE_OAI_API_KEY, // Use environment variable or replace with your key
azureOpenAIApiInstanceName: process.env.AZURE_OAI_API_INSTANCE_NAME, // Use environment variable or replace with your instance name
azureOpenAIApiEmbeddingsDeploymentName: process.env.AZURE_OAI_API_EMBEDDINGS_DEPLOYMENT_NAME, // Use environment variable or replace with your deployment name
azureOpenAIApiVersion: process.env.AZURE_OAI_API_EMBED_VERSION, // Use environment variable or replace with your API version
maxRetries: 1,
});

// Initialize Chat Model with Azure OpenAI API key
const llm = new ChatOpenAI({
  temperature: 0.5,
  azureOpenAIApiKey: process.env.AZURE_OAI_API_KEY, // Use environment variable
  azureOpenAIApiInstanceName: process.env.AZURE_OAI_API_INSTANCE_NAME, // Use environment variable
  azureOpenAIApiVersion: process.env.AZURE_OAI_API_CHAT_VERSION, // Ensure this is correct
  azureOpenAIApiDeploymentName: process.env.AZURE_OAI_CHAT_DEPLOYMENT_NAME, // Ensure this matches Azure deployment
});



const app = express();
const PORT = 5000;
app.use(express.json());
app.use(cors());

  

// MongoDB connection
const mongoURI = process.env.MONGO_URI; // Update with your database name
mongoose.connect(mongoURI).then(() => console.log("MongoDB connected"))
  .catch(err => console.error("MongoDB connection error:", err));

// Define a Mongoose schema for storing chunks and embeddings
const chunkSchema = new mongoose.Schema({
  fileName: { type: String, required: true },
  chunk: { type: String, required: true },
  embedding: { type: [Number], required: true }, // Array of numbers representing the embedding
  createdAt: { type: Date, default: Date.now }
});

const Chunk = mongoose.model("Chunk", chunkSchema);

// Configure multer for file uploads
const upload = multer({ dest: "uploads/" });

// Endpoint to handle file uploads, parsing, embedding, and storing

app.post("/upload", upload.array("files", 10), async (req, res) => {
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: "No files were uploaded." });
        }

        const chunksArray = [];

        // Process each uploaded file
        for (const file of req.files) {
            const filePath = path.join(__dirname, file.path);
            console.log("File path:", filePath);

            let loader;
            let content = "";

            // Step 1: Determine the file type (DOCX or PDF) and use the appropriate loader
            if (path.extname(file.originalname).toLowerCase() === ".docx") {
                loader = new DocxLoader(filePath);
                console.log("Loading DOCX document...");
                const docs = await loader.load();
                if (docs.length > 0) {
                    content = docs[0].pageContent; // Extract text content from the DOCX file
                    console.log("Document content:", content);
                }
            } else if (path.extname(file.originalname).toLowerCase() === ".pdf") {
                console.log("Loading PDF document...");
                const pdfBuffer = fs.readFileSync(filePath);
                const pdfData = await pdfParse(pdfBuffer);
                content = pdfData.text; // Extract text content from the PDF
                console.log("Document content:", content);
            } else {
                // If the file format is not supported, skip it
                console.log("Unsupported file format:", file.originalname);
                continue;
            }

            // Cleanup: delete the uploaded file after processing
            fs.unlinkSync(filePath);
            console.log("File deleted.");

            // Step 2: Use a splitter to break the content into chunks
            const splitter = new RecursiveCharacterTextSplitter({
                chunkSize: 1000, // Define your chunk size here
                chunkOverlap: 200, // Optional overlap between chunks
            });

            console.log("Splitting document...");
            const chunks = await splitter.splitText(content);
            console.log("Chunks:", chunks);

            // Step 3: Embed each chunk using Azure embeddings
            const embeddingsArray = await embeddings.embedDocuments(chunks);
            console.log("Generated embeddings:", embeddingsArray);

            // Store each chunk and its embedding in MongoDB
            for (let i = 0; i < chunks.length; i++) {
                const newChunk = new Chunk({
                    fileName: file.originalname,
                    chunk: chunks[i],
                    embedding: embeddingsArray[i]
                });
                await newChunk.save();
                console.log(`Chunk from ${file.originalname} saved with embedding.`);
            }

            // Push the chunks for this file into the result array
            chunksArray.push({
                fileName: file.originalname,
                chunks,
            });
        }

        // Return all chunks from all files as a JSON response
        res.json({ files: chunksArray });
    } catch (error) {
        console.error("Error processing the documents:", error);
        res.status(500).send("Failed to process the files");
    }
});

// Import cosine similarity function or implement it
const cosineSimilarity = (vecA, vecB) => {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
};

app.post("/chat", async (req, res) => {
  try {
      const userInput = req.body.question;
      console.log("user input:", userInput);

      // Step 1: Embed the user's query
      const queryEmbedding = await embeddings.embedDocuments([userInput]); // Returns an array, so we access [0]
      const queryVector = queryEmbedding[0];
      console.log("Query embedding:", queryVector);

      // Step 2: Retrieve all chunks from MongoDB
      const allChunks = await Chunk.find();
      console.log("Retrieved all chunks for similarity comparison.");

      // Step 3: Compare the query embedding with chunk embeddings using cosine similarity
      const similarChunks = allChunks
          .map(chunk => {
              const similarityScore = cosineSimilarity(queryVector, chunk.embedding);
              return { chunk, similarityScore };
          })
          .sort((a, b) => b.similarityScore - a.similarityScore) // Sort by highest similarity
          .slice(0, 5); // Get the top 5 most similar chunks

      if (similarChunks.length === 0) {
          return res.status(404).json({ error: "No relevant information found." });
      }

      // Step 4: Format the context for the prompt
      const context = similarChunks.map(item => `${item.chunk.fileName}: ${item.chunk.chunk}`).join("\n");
      console.log("context:", context);

// Step 5: Generate a response using the LLM with the retrieved context
const prompt = `Welcome to your intelligent AI companion! I'm here to provide you with insightful and accurate answers to your questions.

**Greeting Response:** If the user says "Hi" or greets me in any way, I will respond with: 
"Hello there! 🌟 I'm delighted to see you! How can I assist you today? Whether you have a question, need information, or just want to chat, I'm here to help. Let's explore together!" 

Here are the relevant excerpts that can help you answer the user's question:

Context:
${context}

**Question:** ${userInput}

In crafting my response, I will:

1. **Deliver Insightful Information:** Provide clear and relevant insights that address your query, ensuring you receive the most accurate information.

2. **Engage Conversationally:** Interact in a friendly and engaging manner, making our conversation enjoyable and informative.

3. **Encourage Exploration:** If there are areas where further information could enhance your understanding, I will suggest additional resources or topics for you to explore.

Let's dive into your question and uncover the knowledge you seek together!`;

      const response = await llm.invoke(prompt);
      console.log("reponse:",response)
      // Step 6: Send the generated response back to the client
      res.json({ answer: response.content });
  } catch (error) {
      console.error("Error processing the chat request:", error);
      res.status(500).send("Failed to process the chat request");
  }
});

// Route to serve a simple message
app.get('/', (req, res) => {
    res.send('Server is running for documents chat !');
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server running on ${PORT}`);
});
