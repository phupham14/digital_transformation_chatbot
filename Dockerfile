FROM python:3.10-slim

WORKDIR /app

# Copy code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy start script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose port (Render sẽ inject PORT nhưng ta dùng default 10000)
EXPOSE 10000

# Start app
CMD ["./start.sh"]