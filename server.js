const express = require("express");
const multer = require("multer");
const { DocxLoader } = require("@langchain/community/document_loaders/fs/docx");
const { PDFLoader } = require("@langchain/community/document_loaders/fs/pdf");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { AzureOpenAIEmbeddings } = require("@langchain/openai");
const fs = require("fs");
const path = require("path");
const mongoose = require("mongoose");
const { ChatOpenAI } = require("@langchain/openai");
const cors = require("cors");
const pdfParse = require("pdf-parse");
const timeout = require('connect-timeout');

require('dotenv').config();

const app = express();
const PORT = 5000;
app.use(express.json());
app.use(cors());
app.use(timeout(1200000));

// MongoDB connection
const mongoURI = process.env.MONGO_URI;
mongoose.connect(mongoURI).then(() => console.log("MongoDB connected"))
    .catch(err => console.error("MongoDB connection error:", err));

// Define a Mongoose schema for storing chunks and embeddings
const chunkSchema = new mongoose.Schema({
    fileName: { type: String, required: true },
    chunk: { type: String, required: true },
    embedding: { type: [Number], required: true },
    createdAt: { type: Date, default: Date.now }
});

const Chunk = mongoose.model("Chunk", chunkSchema);

// Configure multer for file uploads
const upload = multer({
    dest: "uploads/",
    limits: { fileSize: 100 * 1024 * 1024 }
});

// Initialize Azure OpenAI embeddings
const embeddings = new AzureOpenAIEmbeddings({
    azureOpenAIApiKey: process.env.AZURE_OAI_API_KEY,
    azureOpenAIApiInstanceName: process.env.AZURE_OAI_API_INSTANCE_NAME,
    azureOpenAIApiEmbeddingsDeploymentName: process.env.AZURE_OAI_API_EMBEDDINGS_DEPLOYMENT_NAME,
    azureOpenAIApiVersion: process.env.AZURE_OAI_API_EMBED_VERSION,
    maxRetries: 1,
});

// Initialize Chat Model with Azure OpenAI API key
const llm = new ChatOpenAI({
    temperature: 0.5,
    azureOpenAIApiKey: process.env.AZURE_OAI_API_KEY,
    azureOpenAIApiInstanceName: process.env.AZURE_OAI_API_INSTANCE_NAME,
    azureOpenAIApiVersion: process.env.AZURE_OAI_API_CHAT_VERSION,
    azureOpenAIApiDeploymentName: process.env.AZURE_OAI_CHAT_DEPLOYMENT_NAME,
});

// Caching mechanism for embeddings
let embeddingCache = {};

async function getEmbedding(text) {
    if (embeddingCache[text]) {
        return embeddingCache[text];
    }
    
    try {
        const embedding = await embeddings.embedDocuments([text]);
        embeddingCache[text] = embedding[0]; // Cache the result
        return embedding[0];
    } catch (error) {
        if (error.response && error.response.status === 429) {
            console.warn("Rate limit exceeded. Retrying...");
            await new Promise(resolve => setTimeout(resolve, 5000)); // Wait before retrying
            return getEmbedding(text); // Retry embedding
        }
        throw error; // Re-throw if it's not a rate limit issue
    }
}

// Endpoint to handle file uploads, parsing, embedding, and storing
app.post("/upload", upload.array("files", 10), async (req, res) => {
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: "No files were uploaded." });
        }

        const chunksArray = [];

        for (const file of req.files) {
            const filePath = path.join(__dirname, file.path);
            let loader;
            let content = "";

            if (path.extname(file.originalname).toLowerCase() === ".docx") {
                loader = new DocxLoader(filePath);
                const docs = await loader.load();
                content = docs[0].pageContent;
            } else if (path.extname(file.originalname).toLowerCase() === ".pdf") {
                const pdfBuffer = fs.readFileSync(filePath);
                const pdfData = await pdfParse(pdfBuffer);
                content = pdfData.text;
            } else {
                continue; // Skip unsupported formats
            }

            fs.unlinkSync(filePath); // Cleanup

            const splitter = new RecursiveCharacterTextSplitter({
                chunkSize: 1000,
                chunkOverlap: 200,
            });

            const chunks = await splitter.splitText(content);

            // Use batch processing for embeddings
            const embeddingPromises = chunks.map(chunk => getEmbedding(chunk));
            const embeddingsArray = await Promise.all(embeddingPromises);

            for (let i = 0; i < chunks.length; i++) {
                const newChunk = new Chunk({
                    fileName: file.originalname,
                    chunk: chunks[i],
                    embedding: embeddingsArray[i]
                });
                await newChunk.save();
            }

            chunksArray.push({ fileName: file.originalname, chunks });
        }

        res.json({ files: chunksArray });
    } catch (error) {
        console.error("Error processing the documents:", error);
        res.status(500).send("Failed to process the files");
    }
});

// Cosine similarity function
const cosineSimilarity = (vecA, vecB) => {
    let dotProduct = vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
    let normA = Math.sqrt(vecA.reduce((sum, val) => sum + val * val, 0));
    let normB = Math.sqrt(vecB.reduce((sum, val) => sum + val * val, 0));
    
    return dotProduct / (normA * normB);
};

app.post("/chat", async (req, res) => {
    try {
        const userInput = req.body.question;

        const queryVector = await getEmbedding(userInput);

        const allChunks = await Chunk.find();

        const similarChunks = allChunks.map(chunk => ({
            chunk,
            similarityScore: cosineSimilarity(queryVector, chunk.embedding)
        }))
        .sort((a, b) => b.similarityScore - a.similarityScore)
        .slice(0, 5);

        if (similarChunks.length === 0) {
            return res.status(404).json({ error: "No relevant information found." });
        }

        const context = similarChunks.map(item => `${item.chunk.fileName}: ${item.chunk.chunk}`).join("\n");

        const prompt = `Welcome to your intelligent AI companion! I'm here to provide you with insightful and accurate answers to your questions.

Here are the relevant excerpts that can help you answer the user's question:

Documents:
${context}

*Question:* ${userInput}`;

        const responseContent = await llm.invoke(prompt);

        res.json({ answer: responseContent.content });
    } catch (error) {
        console.error("Error processing the chat request:", error);
        res.status(500).send("Failed to process the chat request");
    }
});

// Route to serve a simple message
app.get('/', (req, res) => {
    res.send('Server is running for documents chat!');
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server running on ${PORT}`);
});
