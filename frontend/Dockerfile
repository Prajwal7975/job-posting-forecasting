# ==========================================================
# Production frontend
# ==========================================================

FROM nginx:alpine

# Remove default Nginx website
RUN rm -rf /usr/share/nginx/html/*

# Copy static frontend
COPY index.html /usr/share/nginx/html/

COPY css /usr/share/nginx/html/css
COPY js /usr/share/nginx/html/js

# Custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]