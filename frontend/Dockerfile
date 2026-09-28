FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .

# Vite bakes VITE_* vars into the static bundle at build time. These are backend LOCATION
# config (never secrets - the operator's API key is entered at runtime and lives only in
# sessionStorage), so it is safe for them to end up in the built JS.
ARG VITE_API_BASE_URL=http://localhost:8000
ARG VITE_WS_BASE_URL=ws://localhost:8000
ARG VITE_DEFAULT_CORPUS_ID=northstar_demo_extended
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_WS_BASE_URL=$VITE_WS_BASE_URL
ENV VITE_DEFAULT_CORPUS_ID=$VITE_DEFAULT_CORPUS_ID
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
