const { onRequest } = require("firebase-functions/v2/https");
const { logger } = require("firebase-functions");
const axios = require("axios");

const FASTAPI_BASE_URL = "https://unintruding-overcoyly-peter.ngrok-free.dev";

exports.callAI = onRequest(
  { region: "asia-northeast3" },
  async (req, res) => {
    try {
      const { userId, top_k } = req.body;

      if (!userId) {
        res.status(400).send({ error: "userId is required" });
        return;
      }

      const response = await axios.post(`${FASTAPI_BASE_URL}/recommend`, {
        userId,
        top_k: top_k || 3,
      });

      res.status(200).send(response.data);
    } catch (error) {
      logger.error("AI server error:", error.message);

      if (error.response) {
        res.status(500).send({
          error: "FastAPI response error",
          detail: error.response.data,
        });
      } else {
        res.status(500).send({
          error: "AI server error",
          detail: error.message,
        });
      }
    }
  }
);

