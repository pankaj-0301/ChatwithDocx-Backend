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
const { Readable } = require("stream");

require("dotenv").config();

const embeddings = new AzureOpenAIEmbeddings({
  azureOpenAIApiKey: process.env.AZURE_OAI_API_KEY,
  azureOpenAIApiInstanceName: process.env.AZURE_OAI_API_INSTANCE_NAME,
  azureOpenAIApiEmbeddingsDeploymentName: process.env.AZURE_OAI_API_EMBEDDINGS_DEPLOYMENT_NAME,
  azureOpenAIApiVersion: process.env.AZURE_OAI_API_EMBED_VERSION,
  maxRetries: 1,
});

const llm = new ChatOpenAI({
  temperature: 0.5,
  azureOpenAIApiKey: process.env.AZURE_OAI_API_KEY,
  azureOpenAIApiInstanceName: process.env.AZURE_OAI_API_INSTANCE_NAME,
  azureOpenAIApiVersion: process.env.AZURE_OAI_API_CHAT_VERSION,
  azureOpenAIApiDeploymentName: process.env.AZURE_OAI_CHAT_DEPLOYMENT_NAME,
});

const app = express();
const PORT = 5000;
app.use(express.json());
app.use(cors());

mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log("MongoDB connected"))
  .catch(err => console.error("MongoDB connection error:", err));

const chunkSchema = new mongoose.Schema({
  fileName: { type: String, required: true },
  chunk: { type: String, required: true },
  embedding: { type: [Number], required: true },
  createdAt: { type: Date, default: Date.now }
});

const Chunk = mongoose.model("Chunk", chunkSchema);

const upload = multer({
  dest: "uploads/",
  limits: { fileSize: 100 * 1024 * 1024 }
});

// Helper function to stream content for processing
const streamFileContent = async (filePath, fileType) => {
  return new Promise((resolve, reject) => {
    let content = "";
    const stream = fs.createReadStream(filePath, { encoding: "utf8" });

    stream.on("data", chunk => {
      content += chunk;
    });

    stream.on("end", () => {
      resolve(content);
    });

    stream.on("error", err => {
      reject(err);
    });
  });
};

app.post("/upload", upload.array("files", 10), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: "No files were uploaded." });
    }

    const chunksArray = [];

    for (const file of req.files) {
      const filePath = path.join(__dirname, file.path);
      let content = "";

      if (path.extname(file.originalname).toLowerCase() === ".docx") {
        const loader = new DocxLoader(filePath);
        const docs = await loader.load();
        content = docs.length > 0 ? docs[0].pageContent : "";
      } else if (path.extname(file.originalname).toLowerCase() === ".pdf") {
        const stream = fs.createReadStream(filePath);
        const pdfData = await pdfParse(stream);
        content = pdfData.text;
      } else {
        content = await streamFileContent(filePath);
      }

      fs.unlinkSync(filePath);

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 0,
      });

      const chunks = await splitter.splitText(content);

      const batchSize = 5;

      for (let i = 0; i < chunks.length; i += batchSize) {
        const batchChunks = chunks.slice(i, i + batchSize);
        const embeddingsArray = await embeddings.embedDocuments(batchChunks);

        const chunkDocs = batchChunks.map((chunk, j) => ({
          fileName: file.originalname,
          chunk,
          embedding: embeddingsArray[j]
        }));

        await Chunk.insertMany(chunkDocs);
      }

      chunksArray.push({
        fileName: file.originalname,
        chunks,
      });
    }

    res.json({ files: chunksArray });
  } catch (error) {
    console.error("Error processing the documents:", error);
    res.status(500).send("Failed to process the files");
  }
});

app.post("/chat", async (req, res) => {
  try {
    const userInput = req.body.question;
    const queryEmbedding = await embeddings.embedDocuments([userInput]);
    const queryVector = queryEmbedding[0];

    const cursor = Chunk.find().cursor();

    const similarChunks = [];
    for (let chunk = await cursor.next(); chunk != null; chunk = await cursor.next()) {
      const similarityScore = cosineSimilarity(queryVector, chunk.embedding);
      similarChunks.push({ chunk, similarityScore });
    }

    similarChunks.sort((a, b) => b.similarityScore - a.similarityScore);

    const topChunks = similarChunks.slice(0, 5);
    const context = topChunks.map(item => `${item.chunk.fileName}: ${item.chunk.chunk}`).join("\n");

    const prompt = `Documents:\n${context}\n*Question:* ${userInput}`;

    const response = await llm.invoke(prompt);
    res.json({ answer: response.content });
  } catch (error) {
    console.error("Error processing the chat request:", error);
    res.status(500).send("Failed to process the chat request");
  }
});

app.get('/', (req, res) => {
  res.send('Server is running for documents chat!');
});

app.listen(PORT, () => {
  console.log(`Server running on ${PORT}`);
});

// Utility function to calculate cosine similarity
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
